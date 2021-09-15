def compute_sample_points(ntimes, winsize, stepsize, clip=True):
    sample_points = []
    start_samp = 0
    while start_samp < ntimes:
        end_samp = start_samp + winsize
        if end_samp > ntimes:
            if clip:
                return sample_points
            else:
                end_samp = ntimes
        sample_points.append((start_samp, end_samp))
        start_samp = start_samp + stepsize
    return sample_points
