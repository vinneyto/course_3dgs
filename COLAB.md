# Google Colab + CUDA workflow

This branch keeps source code in GitHub and large datasets/checkpoints in
Google Drive. A Colab VM is treated as disposable compute: every new runtime
clones the branch and mounts the persistent data directory from Drive.

## Open the notebook

[Open `3dgs_colab_cuda.ipynb` in Colab](https://colab.research.google.com/github/vinneyto/course_3dgs/blob/colab-cuda/3dgs_colab_cuda.ipynb)

Before running it, select **Runtime → Change runtime type → T4 GPU** (or another
GPU). The notebook deliberately stops when CUDA is unavailable instead of
silently running the course on CPU.

## Google Drive layout

Upload the course data to:

```text
MyDrive/
└── course_3dgs/
    └── data/
        ├── out_bonsai/
        │   ├── pos_param.pt
        │   ├── alpha_raw_param.pt
        │   ├── f_dc.pt
        │   ├── scale_raw.pt
        │   └── rot_raw.pt
        ├── out_colmap/
        │   └── bonsai/
        │       ├── cam_meta.npy
        │       └── cameras.npy
        └── image_data/
            └── bonsai/
                └── images_2/
```

`f_rest.pt` is optional for the interactive tutorial. The notebook creates
symbolic links from the transient checkout to these three Drive directories;
the data is never copied into GitHub or committed accidentally.

## Synchronization model

- **GitHub → Colab:** restart the runtime and run the setup cells. The checkout
  is cloned fresh, or fast-forwarded when it already exists in the same VM.
- **Google Drive → Colab:** mounting Drive makes datasets and saved outputs
  persistent across runtimes.
- **Notebook edits → GitHub:** use **File → Save a copy in GitHub**, select
  `vinneyto/course_3dgs`, branch `colab-cuda`, and keep the path
  `3dgs_colab_cuda.ipynb`.
- **Python/source edits → GitHub:** edit and commit them in a normal local
  checkout. Avoid putting a long-lived GitHub token directly into a notebook.

Do not store datasets under the Git repository. Colab VMs are temporary, and
anything left only under `/content` disappears when the runtime is recycled.
