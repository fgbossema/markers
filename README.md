# Markers

Using markers and post-processing to create 3D CT images with 2D radiography setups. See Bossema et al. 2024 for more information. 

* Free software: GNU General Public License v3



## Readiness

The author of this package is in the process of setting up this
package for optimal usability. The following has already been completed:

- [ ] Documentation
    - A package description has been written in the README
    - Documentation has been generated using `make docs`, committed,
        and pushed to GitHub.
	- GitHub pages have been setup in the project settings
	  with the "source" set to "master branch /docs folder".
- [ ] An initial release
	- In `CHANGELOG.md`, a release date has been added to v0.1.0 (change the YYYY-MM-DD).
	- The release has been marked a release on GitHub.
	- For more info, see the [Software Release Guide](https://cicwi.github.io/software-guides/software-release-guide).
- [x] A conda environment yml has been added. 
	

## Getting Started

It takes a few steps to setup Markers on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.

### Installing with yml
Create a conda environment by running ‘conda env create -f markers.yml’. Note: this can take a while (5-10mins).
Activate the environment (‘conda activate markers’).

### Installing from source

To install Markers, simply clone this GitHub
project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/fgbossema/markers.git
cd markers
pip install -e .
```

## Authors and contributors

* **Francien Bossema** - *Initial work*
* **Willem Jan Palenstijn**
* **Robert van Liere** - *Inpainting code*

See also the list of [contributors](https://github.com/fgbossema/markers/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `main` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
