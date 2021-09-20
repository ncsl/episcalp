from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from eztrack.fragility import state_lds_array
import matplotlib

from episcalp import read_scalp_eeg

matplotlib.use('Qt5Agg')
from mne_bids import get_entity_vals, BIDSPath, read_raw_bids


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
    nConn = int(np.round(nCh * percConn))

    A_abs = np.absolute(A_mat)
    np.fill_diagonal(A_abs, 0)

    # Identify sources and sinks for each channel
    ch_influenced_ind = np.empty([nCh, nConn])
    ch_influential_ind = np.empty([nCh, nConn])

    for chdx in range(nCh):
        A_row = A_abs[chdx, :]  # ensure self connection gets lowest rank
        A_col = A_abs[:, chdx]  # ensure self connection gets lowest rank

        sort_ch_r = np.argsort(A_row, axis=0)[::-1]
        sort_ch_c = np.argsort(A_col, axis=0)[::-1]

        ch_influential_ind[chdx, :] = [int(sc) for sc in sort_ch_r[0:nConn]]
        ch_influenced_ind[chdx, :] = [int(sc) for sc in sort_ch_c[0:nConn]]
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

    A_abs = np.absolute(A_mat)
    np.fill_diagonal(A_abs, 0)

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

    nSS = int(np.round(nCh*percSS))
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
        deriv_path=None,
        savename=None

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
    nSinks = len(sink_ind)
    nSources = len(source_ind)

    others_ind = [ch for ch in range(nCh) if ch not in source_ind+sink_ind]
    others_ch = chnames[others_ind]

    ch_influential_ind, ch_influenced_ind = _identify_influence_channels(A_mean, thresh_percConn)

    sink_influenced_ind = [int(cii[0]) for cii in ch_influenced_ind[sink_ind,:]]
    sink_influenced_ch = chnames[sink_influenced_ind]

    source_influenced_ind = [int(cii[0]) for cii in  ch_influenced_ind[source_ind, :]]
    source_influenced_ch = chnames[source_influenced_ind]

    fig, ax = plt.subplots()
    ax.scatter(p_row_ranks[sink_ind], p_col_ranks[sink_ind], 10, 'purple')
    ax.scatter(p_row_ranks[source_ind], p_col_ranks[source_ind], 10, 'red')
    ax.scatter(p_row_ranks[others_ind], p_col_ranks[others_ind], 10, 'gray')
    for ind, (r,c) in enumerate(zip(p_row_ranks, p_col_ranks)):
        ch_name = chnames[ind]
        ax.text(r+0.01, c+0.01, ch_name)

    ax.set_xlabel('Influence from others')
    ax.set_ylabel('Influence to others')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    quiver_units = {
        "width": 0.0005,
        "head_width": 0.005,
    }

    for idx, s_ind in enumerate(source_ind):
        source_infl_ind = np.array(source_influenced_ind[idx])

        source_x = np.tile(p_row_ranks[s_ind], source_infl_ind.shape)
        source_y = np.tile(p_col_ranks[s_ind], source_infl_ind.shape)
        infl_x = p_row_ranks[source_infl_ind]
        infl_y = p_col_ranks[source_infl_ind]

        sDiff_x = infl_x - source_x
        sDiff_y = infl_y - source_y

        print(source_x, source_y)
        print(sDiff_x, sDiff_y)

        #ax.arrow(source_x, source_y, sDiff_x, sDiff_y, **quiver_units)

    for idx, s_ind in enumerate(sink_ind):
        sink_infl_ind = np.array(sink_influenced_ind[idx])

        sink_x = np.tile(p_row_ranks[s_ind], sink_infl_ind.shape)
        sink_y = np.tile(p_col_ranks[s_ind], sink_infl_ind.shape)
        infl_x = p_row_ranks[sink_infl_ind]
        infl_y = p_col_ranks[sink_infl_ind]

        sDiff_x = infl_x - sink_x
        sDiff_y = infl_y - sink_y

        #ax.arrow(sink_x, sink_y, sDiff_x, sDiff_y, **quiver_units)

    #plt.show(block=True)
    plt.savefig(savename)


if __name__ == "__main__":
    """
    edf_fpath = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/bids/derivatives/ICA/1-30Hz-30/win-20/sub-jhh003/eeg/sub-jhh003_run-01_eeg.edf")
    raw = mne.io.read_raw_edf(edf_fpath)
    X = raw.get_data()

    model_params = {
        "winsize": 100,
        "stepsize": 100,
        "l2penalty": 0,
        "method_to_use": 'pinv',
        "n_jobs": 1,
    }

    A_mats = state_lds_array(X, **model_params)
    ch_names = np.array(raw.ch_names)
    sf = raw.info['sfreq']
    plot_2d_ss_space(A_mats, ch_names, sf)"""

    root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/bids/derivatives/ICA/1-30Hz-30/win-20")
    deriv_root = Path("D:/OneDriveParent/OneDrive - Johns Hopkins/Shared Documents/bids/derivatives")
    figures_root = deriv_root
    figure_ext = ".png"

    subjects = get_entity_vals(root, 'subject')
    for subject in subjects:
        ignore_subjects = [sub for sub in subjects if sub is not subject]
        sessions = get_entity_vals(root, 'session', ignore_subjects=ignore_subjects)
        if len(sessions) == 0:
            sessions = [None]
        for session in sessions:
            ignore_sessions = [ses for ses in sessions if ses is not session]
            tasks = get_entity_vals(root, 'task', ignore_subjects=ignore_subjects, ignore_sessions=ignore_sessions)
            if len(tasks) == 0:
                tasks = [None]
            for task in tasks:
                ignore_tasks = [tsk for tsk in tasks if tsk is not task]
                runs = get_entity_vals(root, 'run', ignore_subjects=ignore_subjects, ignore_sessions=ignore_sessions,
                                       ignore_tasks=ignore_tasks)
                for run in runs:
                    bids_params = {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "run": run,
                        "datatype": "eeg",
                        "extension": ".edf",
                    }
                    bids_path = BIDSPath(root=root, **bids_params)
                    raw = read_raw_bids(bids_path)
                    sf = raw.info['sfreq']
                    resample_sfreq=None
                    if sf > 256:
                        resample_sfreq = 256
                    raw = read_scalp_eeg(bids_path, reference=None, resample_sfreq=resample_sfreq)
                    X = raw.get_data()

                    model_params = {
                        "winsize": 100,
                        "stepsize": 100,
                        "l2penalty": 0,
                        "method_to_use": 'pinv',
                        "n_jobs": 1,
                    }

                    A_mats = state_lds_array(X, **model_params)
                    ch_names = np.array(raw.ch_names)

                    savename = figures_root / "figures" / "sourcesink" / bids_path.basename.replace(".edf", "_2dplot.png")
                    plot_2d_ss_space(A_mats, ch_names, sf, deriv_path=deriv_root, savename=savename)

