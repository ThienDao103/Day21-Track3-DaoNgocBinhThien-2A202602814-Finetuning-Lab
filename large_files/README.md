# Large lab artifacts

This repository is a public fork. Its GitHub remote refused new Git LFS uploads.
The original large artifacts are therefore stored here in chunks below 50 MiB.
The manifest records each chunk and each original file's size and SHA-256.

After cloning, restore the exact original files:

```bash
python scripts/restore_large_files.py
```

This restores the evaluated adapter weights, the imported adapter ZIP and the final
submission ZIP. It does not quantize, retrain or modify the measured artifacts.
Run restoration before evaluating the adapter or running the submission validator.
