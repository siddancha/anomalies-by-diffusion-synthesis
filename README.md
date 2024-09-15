# Diffusion-based uncertainty

## Install instructions

1. Install [Git LFS](https://git-lfs.com/).

2. Clone repository and `cd` into it.
   ```bash
   git clone --recursive https://github.com/siddancha/diffunc.git
   cd /path/to/diffunc
   ```

3. Install `mamba` by following the [official install instructions](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

4. Create a mamba environment with Python 3.10.
   ```bash
   mamba create -n DUMamba python==3.10
   ```

5. Activate mamba environment
   ```bash
   mamba activate DUMamba
   ```

6. Install CUDA in the mamba environment.
   ```bash
   mamba install -c nvidia/label/cuda-12.4.0 cuda
   ```
   Make sure CUDA is installed by running `nvcc --version`. You should see some output saying that you have CUDA 12.4 installed.

7. Create a new Python virtual environment
   ```bash
   python3 -m venv .venv --prompt=DUVenv
   ```

8. Activate the virtual environment
   ```bash
   source .venv/bin/activate
   ```

9. Install [pdm](https://pdm.fming.dev/)
   ```bash
   pip install pdm
   ```

10. Install Python dependencies via pdm.
    ```bash
    pdm install -v
    ```
    - On MIT Supercloud, you may need to run this:
      ```bash
      TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" pdm install -v
      ```
      or add `export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"` to your `.bashrc`.
12. Create the following symlink for `featup`:
    ```bash
    ln -s `pwd`/.venv/lib/python3.10/site-packages/clip/bpe_simple_vocab_16e6.txt.gz .venv/lib/python3.10/site-packages/featup/featurizers/maskclip/
    ```

## Setting up the RUGD dataset
1. Create a folder (or symlinked folder) called `data` inside the `diffunc` repo.
   ```bash
   mkdir data
   ```

2. Download the [RUGD dataset](http://rugd.vision/).

3. Unzip the downloaded files and structure the dataset as follows:
   ```
   data/RUGD
   ├── RUGD_frames-with-annotations
         ├── creek, park-1, etc.          -- folders for each scene containing ".png" files from the RGB camera.
   ├── RUGD_annotations
         ├── creek, park-1, etc.          -- folders for each scene containing ".png" label color images, colored using the class palette.
         ├── RUGD_annotation-colormap.txt -- mapping containing a list of class id, class name and class color.
   ```

4. Run the dataset conversion scripts.
   ```bash
   ./scripts/make_ddpm_train_set_rugd_full.py
   ./scripts/make_ddpm_train_set_rugd_trail_trail_15.py
   ```

## Working with docker image
<details>
<summary>Click here to expand</summary><br>

### Docker commands
- To build the docker image under the `ppc64le` platform:
   ```bash
   docker build --platform ppc64le --tag jammy-python .
   ```
- To run the docker container:
   ```bash
   docker run --rm -i -t --platform ppc64le \
       -v /home/sancha/container/envs:/root/envs \
       -v /home/sancha/repos:/root/repos \
       jammy-python:latest /bin/bash
   ```
</details>
