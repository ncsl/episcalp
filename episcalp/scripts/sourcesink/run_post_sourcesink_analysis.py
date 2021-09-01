from sample_code.study import generate_patient_features


def run_post_analysis(deriv_path=None, subject=None, features=None):
    if subject is not None:
        subjects = [subject]
    else:
        subjects = None
    generate_patient_features(
        deriv_path, "sourcesink", features, subjects=subjects, verbose=True
    )
