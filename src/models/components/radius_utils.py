import numpy as np
import torch
from torch_scatter import segment_coo, segment_csr


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res

# Borrowed from GemNet.
def select_symmetric_edges(tensor, mask, reorder_idx, inverse_neg):
    # Mask out counter-edges
    tensor_directed = tensor[mask]
    # Concatenate counter-edges after normal edges
    sign = 1 - 2 * inverse_neg
    tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
    # Reorder everything so the edges of every image are consecutive
    tensor_ordered = tensor_cat[reorder_idx]
    return tensor_ordered
    
def symmetrize_edges(
    edge_index,
    cell_offsets,
    neighbors,
):
    # Generate mask
    mask_sep_atoms = edge_index[0] < edge_index[1]
    # Distinguish edges between the same (periodic) atom by ordering the cells
    cell_earlier = (
        (cell_offsets[:, 0] < 0)
        | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
        | (
            (cell_offsets[:, 0] == 0)
            & (cell_offsets[:, 1] == 0)
            & (cell_offsets[:, 2] < 0)
        )
    )
    mask_same_atoms = edge_index[0] == edge_index[1]
    mask_same_atoms &= cell_earlier
    mask = mask_sep_atoms | mask_same_atoms

    # Mask out counter-edges
    edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

    # Concatenate counter-edges after normal edges
    edge_index_cat = torch.cat([edge_index_new, edge_index_new.flip(0)], dim=1)

    # Count remaining edges per image
    batch_edge = torch.repeat_interleave(torch.arange(neighbors.size(0), device=edge_index.device), neighbors)
    batch_edge = batch_edge[mask]
    # segment_coo assumes sorted batch_edge
    # Factor 2 since this is only one half of the edges
    ones = batch_edge.new_ones(1).expand_as(batch_edge)
    neighbors_per_image = 2 * segment_coo(ones, batch_edge, dim_size=neighbors.size(0))

    # Create indexing array
    edge_reorder_idx = repeat_blocks(
        torch.div(neighbors_per_image, 2, rounding_mode="floor"),
        repeats=2,
        continuous_indexing=True,
        repeat_inc=edge_index_new.size(1),
    )

    # Reorder everything so the edges of every image are consecutive
    edge_index_new = edge_index_cat[:, edge_reorder_idx]
    cell_offsets_new = select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)

    return edge_index_new, cell_offsets_new, neighbors_per_image

def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
            max_num_neighbors <= max_num_neighbors_threshold
            or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
            index * max_num_neighbors
            + torch.arange(len(index), device=device)
            - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
                distance_sort[:, max_num_neighbors_threshold]
                + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(
            max=num_included_per_atom
        )

        num_neighbors_image = segment_csr(
            num_neighbors_thresholded, image_indptr
        )

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image

def radius_graph_determinstic(
    data,
    radius,
    max_num_neighbors_threshold,
    symmetrize: bool = True,
    enforce_max_neighbors_strictly: bool = False,
):
    device = data.pos.device

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(num_atoms_per_image, num_atoms_per_image_sqr)

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms.
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr)
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = (torch.arange(num_atom_pairs, device=device) - index_sqr_offset)

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor") + index_offset_expand
    index2 = atom_count_sqr % num_atoms_per_image_expand + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))
    
    if symmetrize:
    
        edge_index, unit_cell, num_neighbors_image = symmetrize_edges(
            edge_index,
            torch.zeros(edge_index.shape[1], 3, device=edge_index.device),
            num_neighbors_image,
        )

    return edge_index, num_neighbors_image # GemNet uses num_neighbors_image

def radius_determinstic(
    data_x,
    data_y,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
):
    device = data_x.pos.device

    # position of the atoms
    atom_pos = data_x.pos
    mesh_pos = data_y.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data_x.natoms
    num_meshs_per_image = data_y.nmeshs
    num_atoms_per_image_sqr = (num_atoms_per_image * num_meshs_per_image).long()

    # atom index offset between images
    atom_index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    atom_index_offset_expand = torch.repeat_interleave(atom_index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(num_atoms_per_image, num_atoms_per_image_sqr)
    
    # mesh index offset between images
    mesh_index_offset = torch.cumsum(num_meshs_per_image, dim=0) - num_meshs_per_image
    mesh_index_offset_expand = torch.repeat_interleave(mesh_index_offset, num_atoms_per_image_sqr)

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms.
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr)
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = (torch.arange(num_atom_pairs, device=device) - index_sqr_offset)

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor") + mesh_index_offset_expand
    index2 = atom_count_sqr % num_atoms_per_image_expand + atom_index_offset_expand

    mesh_pos1 = torch.index_select(mesh_pos, 0, index1)
    atom_pos2 = torch.index_select(atom_pos, 0, index2)

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((mesh_pos1 - atom_pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data_y.nmeshs,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    return edge_index

def get_distances(
    edge_index,
    pos2, # src pos
    pos1=None, # dst pos
    return_distance_vec=True,
):
    j, i = edge_index
    
    if pos1 is None:
        pos1 = pos2
        
    distance_vec = pos2[j] - pos1[i]

    edge_dist = distance_vec.norm(dim=-1)
    
    distance_vec = distance_vec / edge_dist.view(-1, 1)
    
    if return_distance_vec:
        return edge_dist, distance_vec
    else:
        return edge_dist

def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    symmetrize: bool = True,
    pbc=[True, True, True],
):
    device = data.pos.device
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(num_atoms_per_image, num_atoms_per_image_sqr)

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms.
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor") + index_offset_expand
    index2 = atom_count_sqr % num_atoms_per_image_expand + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms_per_image_sqr, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3))
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3))
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))
    
    if symmetrize:
    
        edge_index, unit_cell, num_neighbors_image = symmetrize_edges(
            edge_index,
            unit_cell,
            num_neighbors_image,
        )

    return edge_index, unit_cell, num_neighbors_image # GemNet uses num_neighbors_image

def radius_pbc(
    data_x,
    data_y,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=[True, True, True],
):
    device = data_x.pos.device
    batch_size = len(data_x.natoms)

    if hasattr(data_x, "pbc"):
        data_x.pbc = torch.atleast_2d(data_x.pbc)
        for i in range(3):
            if not torch.any(data_x.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data_x.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data_x.pos
    mesh_pos = data_y.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data_x.natoms
    num_meshs_per_image = data_y.nmeshs
    num_atoms_per_image_sqr = (num_atoms_per_image * num_meshs_per_image).long()

    # atom index offset between images
    atom_index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    atom_index_offset_expand = torch.repeat_interleave(atom_index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(num_atoms_per_image, num_atoms_per_image_sqr)
    
    # mesh index offset between images
    mesh_index_offset = torch.cumsum(num_meshs_per_image, dim=0) - num_meshs_per_image
    mesh_index_offset_expand = torch.repeat_interleave(mesh_index_offset, num_atoms_per_image_sqr)

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms.
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor") + mesh_index_offset_expand
    index2 = atom_count_sqr % num_atoms_per_image_expand + atom_index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(mesh_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data_x.cell[:, 1], data_x.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data_x.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data_x.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data_x.cell[:, 2], data_x.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data_x.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data_x.cell[:, 0], data_x.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data_x.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data_x.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms_per_image_sqr, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3))
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data_y.nmeshs,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3))
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image

def get_distances_pbc(
    edge_index,
    cell,
    cell_offsets,
    num_neighbors_image,
    pos2, # src pos
    pos1=None, # dst pos
    return_distance_vec=False,
):
    j, i = edge_index

    if pos1 is None:
        pos1 = pos2
        
    distance_vectors = pos2[j] - pos1[i]

    # correct for pbc
    neighbors = num_neighbors_image.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]
    distance_vectors = distance_vectors[nonzero_idx]
    
    distance_vectors = distance_vectors / distances.view(-1, 1)
    
    if return_distance_vec:
        return distances, distance_vectors
    else:
        return distances