import numpy as np
import matplotlib.pyplot as plt


def _identify_influence_channels(A_mat, percConn):
    """
    Identify influential channels from A matrix for each channel.

    Most influential channels defined based on highest values of mean(Abs(A)).

    Parameters
    ----------
    A_mat :
    percConn :

    Returns
    -------

    """

    nCh = A_mat.shape[0]
    nConn = np.round(nCh * percConn)

    A_abs = np.abs(A_mat)
    A_abs = np.fill_diagonal(A_abs, 0)

    # Identify sources and sinks for each channel
    ch_influenced_ind = np.empty((nCh, nConn))
    ch_influential_ind = np.empty((nCh, nConn))

    for chdx in range(nCh):
        A_row = A_abs[chdx, :]  # ensure self connection gets lowest rank
        A_col = A_abs[:, chdx]  # ensure self connection gets lowest rank

        sort_ch_r = np.argsort(A_row, axis=0)[::-1]
        sort_ch_c = np.argsort(A_col, axis=0)[::-1]

        ch_influential_ind[chdx, :] = sort_ch_r[0:nConn-1]
        ch_influenced_ind[chdx, :] = sort_ch_c[0:nConn-1]
    return ch_influenced_ind, ch_influenced_ind


def _identifySS_plotRanks(A_mat, percSS):
    """
    Identify sources and sinks from A matrix.

    Parameters
    ----------
    A_mat :
    percSS :

    Returns
    -------

    """

    nCh = A_mat.shape[0]

    A_abs = np.abs(A_mat)
    A_abs = np.fill_diagonal(A_abs, 0)

    # Compute row and column sums
    sum_A_r = np.transpose(np.sum(A_abs, 1))
    sum_A_c = np.sum(A_abs, 0)

    # Identify sources and sinks
    # Rank the channels from lowest (rank 1) to highest (rank nCh) based on row sum.
    # Rank the channels from highest (rank 1) to lowest (rank nCh) based on column sum.
    # Add the two ranks. Sinks = high rank sum and sources = low rank sum.

    sort_ch_r = np.argsort(sum_A_r)
    row_ranks = np.argsort(sort_ch_r)
    row_ranks = np.divide(row_ranks, nCh)

    sort_ch_c = np.argsort(sum_A_c)
    col_ranks = np.argsort(sort_ch_c)
    col_ranks = np.divide(col_ranks, nCh)

    rank_sum = row_ranks + col_ranks
    ss_ind_sort = np.argsort(rank_sum)

    nSS = np.rount(nCh*percSS)
    source_ind = ss_ind_sort[0:nSS-1]
    sink_ind = ss_ind_sort[:-nSS]

    sort_ch_r = np.argsort(sum_A_r)
    p_row_ranks = np.argsort(sort_ch_r)
    p_row_ranks = np.divide(p_row_ranks, nCh)

    sort_ch_c = np.argsort(sum_A_c)
    p_col_ranks = np.argsort(sort_ch_c)
    p_col_ranks = np.divide(p_col_ranks, nCh)
    return row_ranks, col_ranks, rank_sum, ss_ind_sort, source_ind, sink_ind, p_row_ranks, p_col_ranks


def plot_2d_ss_space(
        A_mat,
        chnames,
        fs,
        winsize=0.5,
        stepsize=0.5,
        thresh_percSS=0.1,
        thresh_percConn=0.05,

):
    winsize = winsize*fs
    stepsize = stepsize*fs

    A_mean = np.squeeze(np.mean(A_mat, 2))

    nWin = A_mat.shape[2]
    nCh = len(chnames)
    ez_chs = []

    row_ranks, col_ranks, rank_sum, ss_ind_sort, source_ind, sink_ind, p_row_ranks, p_col_ranks = _identifySS_plotRanks(A_mean, thresh_percSS)

    sink_ch = chnames[sink_ind]
    source_ch = chnames[source_ind]
    nSinks = sink_ind.shape[1]
    nSources = source_ind.shape[1]

    others_ind = [ch for ch in range(nCh) if ch not in source_ind+sink_ind]
    others_ch = chnames[others_ind]

    ch_influential_ind, ch_influenced_ind = _identify_influence_channels(A_mean, thresh_percConn)

    sink_influenced_ind = ch_influenced_ind[sink_ind,:]
    sink_influenced_ch = chnames[sink_influenced_ind]

    source_influenced_ind = ch_influenced_ind[source_ind, :]
    source_influenced_ch = chnames[source_influenced_ind]

    fig, ax = plt.subplots()
    ax.scatter(p_row_ranks[sink_ind], p_col_ranks[sink_ind], 10, 'purple')
    ax.scatter(p_row_ranks[source_ind], p_col_ranks[source_ind], 10, 'red')
    ax.scatter(p_row_ranks[others_ind], p_col_ranks[others_ind], 10, 'gray')

    ax.set_xlabel('Influence from others')
    ax.set_ylabel('Influence to others')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    for idx, s_ind in enumerate(source_ind):
        source_infl_ind = source_influenced_ind[idx]

        source_x = np.matlib.repmat(p_row_ranks[s_ind], source_infl_ind.shape)
        source_y = np.matlib.repmat(p_col_ranks[s_ind], source_infl_ind.shape)
        infl_x = p_row_ranks[source_infl_ind]
        infl_y = p_col_ranks[source_infl_ind]

        sDiff_x = infl_x - source_x
        sDiff_y = infl_y - source_y

        ax.quiver(source_x, source_y, sDiff_x, sDiff_y)

    for idx, s_ind in enumerate(sink_ind):
        sink_infl_ind = sink_influenced_ind[idx]

        sink_x = np.matlib.repmat(p_row_ranks[s_ind], sink_infl_ind.shape)
        sink_y = np.matlib.repmat(p_col_ranks[s_ind], sink_infl_ind.shape)
        infl_x = p_row_ranks[sink_infl_ind]
        infl_y = p_col_ranks[sink_infl_ind]

        sDiff_x = infl_x - sink_x
        sDiff_y = infl_y - sink_y

        ax.quiver(sink_x, sink_y, sDiff_x, sDiff_y)

