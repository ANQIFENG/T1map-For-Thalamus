def download_data():
    from HD_BET.utils import maybe_download_parameters

    for i in range(5):
        maybe_download_parameters(i)


def set_paths():
    from pathlib import Path
    import site

    site_packages_locs = [Path(path) for path in site.getsitepackages()]
    for site_packages_path in site_packages_locs:
        if (site_packages_path / "HD_BET").exists():
            (site_packages_path / "HD_BET" / "paths.py").write_text(
                "folder_with_parameter_files = '/opt/hd-bet_params'\n"
            )
            break


if __name__ == "__main__":
    set_paths()
    download_data()