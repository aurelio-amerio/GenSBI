import glob
import os

files = glob.glob("docs/examples/*.py")

for filepath in files:
    with open(filepath, "r") as f:
        content = f.read()

    # Check if the file has the pattern we want to fix
    if '# os.environ["JAX_PLATFORMS"] = "cuda"' in content and 'import os  # isort: skip' not in content:
        # Find the imports block start
        if "# %% Imports" in content:
            lines = content.splitlines()
            new_lines = []

            # remove the os.environ part from the bottom or wherever it is
            environ_lines = [
                '# Set JAX backend (use \'cuda\' for GPU, \'cpu\' otherwise)',
                '# os.environ["JAX_PLATFORMS"] = "cuda"'
            ]

            # Filter out the environ lines from original location
            lines = [line for line in lines if line not in environ_lines]

            # Reconstruct content

            # Insert at the top after # %% Imports
            for i, line in enumerate(lines):
                new_lines.append(line)
                if line.startswith("# %% Imports"):
                    # check if import os is already there
                    if "import os" in lines[i+1]:
                        # Modify import os line
                        pass # handled later
                    else:
                        # inject import os if not present (unlikely for these files)
                        pass

            # Actually, let's just use string replacement if the structure is consistent
            pass

# Rethinking: string replacement is safer if the structure is exactly as observed.
# The previous ISORT run moved import os to the top group.
# The comments were likely left behind at the bottom of the import block.

# Pattern to replace:
# # %% Imports
# import os
#
# import grain
# ...
# ...
# # Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
# # os.environ["JAX_PLATFORMS"] = "cuda"

# New pattern:
# # %% Imports
# import os  # isort: skip
#
# # Set JAX backend (use 'cuda' for GPU, 'cpu' otherwise)
# # os.environ["JAX_PLATFORMS"] = "cuda"
#
# import grain
# ...
