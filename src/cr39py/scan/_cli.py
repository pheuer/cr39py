"""
Utilities for taking user input
"""

import numpy as np
from IPython import display

from cr39py.scan.base_scan import Scan
from cr39py.scan.cut import Cut
from cr39py.scan.subset import Subset


def _cli_input(
    mode="alphanumeric list", always_pass=None, valid_keys=None, valid_values=None
):  # pragma: no cover
    """
    Collects CLI input from the user: continues asking until a valid
    input has been provided. Input modes:

    'numeric'
        Single number

    'alphanumeric list'
        List of alphanumeric characters, separated by commas

    'key:value list' -> e.g. xmin:10, ymin:-1
        Alternating alpha and numeric key value pairs, comma separated
        Letters are only acceptable in the values if they are 'none'


    always_pass : list of str, optional
        Strings to always accept, regardless of the mode chosen. Usually control
        words like 'end' or 'help'. If not provided, a default list will be
        used.


    valid_keys : list of str, optional
        List of valid keys for key:value list input. Default is to accept
        any key. Keys should all be lowercase


    """
    yesno = set("yn")
    integers = set("1234567890+-")
    floats = integers.union(".e")
    alphas = set("abcdefghijklmnopqrstuvwxyz_-")

    # If the input matches one of these
    if always_pass is None:
        always_pass = ["help", "end"]
    elif always_pass is False:
        always_pass = []

    while True:
        x = str(input(">"))

        if x in always_pass:
            return x

        if mode == "integer":
            if set(x).issubset(integers):
                return int(x)

        elif mode == "float":
            if set(x).issubset(floats):
                return float(x)

        elif mode == "alpha-integer":
            if set(x).issubset(integers.union(alphas)):
                return x

        elif mode == "yn":
            if set(x).issubset(yesno):
                return x

        elif mode == "alpha-integer list":
            split = x.split(",")
            split = [s.strip() for s in split]
            # Discard empty strings
            split = [s for s in split if s != ""]
            if all([set(s).issubset(alphas.union(integers)) for s in split]):
                return split

        elif mode == "key:value list":
            split = x.split(",")
            split = [s.split(":") for s in split]

            # Verify that we have a list of at least one pair, and only pairs
            if all([len(s) == 2 for s in split]) and len(split) > 0:
                # Discard empty strings
                split = [s for s in split if (s[0] != "" and s[1] != "")]

                # Transform any 'none' values into None
                # Strip any other values
                for i, s in enumerate(split):
                    if str(s[1]).lower() == "none":
                        split[i][1] = None
                    else:
                        split[i][1] = s[1].strip()

                # Strip keys, convert to lowercase
                for i in range(len(split)):
                    split[i][0] = split[i][0].strip().lower()

                # Test that values are in the correct sets
                test1 = all(
                    [
                        (
                            (set(s[0].strip()).issubset(alphas))
                            and (s[1] is None or set(s[1]).issubset(floats))
                        )
                        for s in split
                    ]
                )

                # Test that all keys are in the allowed list (if set)
                if valid_keys is not None:
                    test2 = all(s[0] in valid_keys for s in split)
                else:
                    test2 = True

                # Convert any non-None values into floats
                for i, s in enumerate(split):
                    if s[1] is not None:
                        split[i][1] = float(s[1])

                if all([test1, test2]):
                    return {str(s[0].strip()): s[1] for s in split}

                if not test2:
                    print("Key not recognized")

        else:
            raise ValueError("Invalid Mode")


def scan_cli(scan: Scan) -> bool:  # pragma: no cover
    # This flag keeps track of whether any changes have been made
    # by the CLI, and will be returned when it exits
    changed = False

    while True:
        # Clear IPython output to avoid piling up plots
        display.clear_output(wait=False)

        # Create a cut plot
        scan.cutplot(show=True)

        print("*********************************************************")
        print(
            f"Current subset index: {scan._current_subset_index} of {np.arange(len(scan._subsets))}"
        )
        # Print a summary of the current subset
        print(scan.current_subset)
        print(
            f"ntracks selected: {scan.nselected_tracks:.1e} " f"(of {scan.ntracks:.1e})"
        )

        print(
            "add (a), edit (e), edit the domain (d), remove (r), plot (p), "
            "plot inverse (pi), switch subsets (subset), change dslices (dslice), "
            "change the number of dslices (ndslices), end (end), help (help)"
        )

        split = _cli_input(mode="alpha-integer list", always_pass=[])
        x = split[0]

        if x == "help":
            print(
                "Enter commands, followed by any additional arguments "
                "separated by commas.\n"
                " ** Commands ** \n"
                "'a' -> create a new cut\n"
                "'c' -> Select a new dslice\n"
                "Argument (one int) is the index of the dslice to select"
                "Enter 'all' to select all"
                "'d' -> edit the domain\n"
                "'e' -> edit a cut\n"
                "Argument (one int) is the cut to edit\n"
                "'ndslices' -> Change the number of dslices on this subset."
                "'p' -> plot the image with current cuts\n"
                "'pi' -> plot the image with INVERSE of the cuts\n"
                "'r' -> remove an existing cut\n"
                "Arguments are numbers of cuts to remove\n"
                "'subset' -> switch subsets or create a new subset\n"
                "Argument is the index of the subset to switch to, or"
                "'new' to create a new subset"
                "'help' -> print this documentation\n"
                "'end' -> accept the current values\n"
                "'framesize` -> Change the framesize on an axis\n"
                " ** Cut keywords ** \n"
                "xmin, xmax, ymin, ymax, dmin, dmax, cmin, cmax, emin, emax\n"
                "e.g. 'xmin:0,xmax:5,dmax=15'\n"
            )

        elif x == "end":
            scan.cutplot(show=True)
            break

        elif x == "a":
            print("Enter new cut parameters as key:value pairs separated by commas")
            kwargs = _cli_input(mode="key:value list")

            # validate the keys are all valid dictionary keys
            valid = True
            for key in kwargs.keys():
                if key not in list(Cut.defaults.keys()):
                    print(f"Unrecognized key: {key}")
                    valid = False

            if valid:
                c = Cut(**kwargs)
                scan.current_subset.add_cut(c)

            scan.cutplot(show=True)
            changed = True

        elif x == "framesize":
            print("Enter the name of the axis to change")
            ax_name = _cli_input(mode="alpha-integer")
            ax_name = ax_name.upper()
            print(f"Selected axis {ax_name}")
            print(f"Current framesize is {scan._axes[ax_name].framesize:.1e}")
            print("Enter new framesize")
            framesize = _cli_input(mode="float")
            scan.set_framesize(ax_name, framesize)
            scan.cutplot(show=True)
            changed = True

        elif x == "dslice":
            if len(split) < 2:
                print(
                    "Select the index of the dslice to switch to, or"
                    "enter 'all' to select all dslices"
                )
                ind = _cli_input(mode="alpha-integer")
            else:
                ind = split[1]

            if ind == "all":
                scan.select_dslice(None)
            else:
                scan.select_dslice(int(ind))
            scan.cutplot(show=True)
            changed = True

        elif x == "d":
            print("Current domain: " + str(scan.current_subset.domain))
            print(
                "Enter a list key:value pairs with which to modify the domain"
                "(set a key to 'None' to remove it)"
            )
            kwargs = _cli_input(mode="key:value list")
            scan.current_subset.domain.update(**kwargs)
            scan.cutplot(show=True)
            changed = True

        elif x == "e":
            if len(split) > 1:
                ind = int(split[1])

                if ind >= len(scan.current_subset.cuts):
                    print("Invalid subset number")

                else:
                    print(
                        f"Selected cut ({ind}) : " + str(scan.current_subset.cuts[ind])
                    )
                    print(
                        "Enter a list key:value pairs with which to modify this cut"
                        "(set a key to 'None' to remove it)"
                    )

                    kwargs = _cli_input(mode="key:value list")
                    scan.current_subset.cuts[ind].update(**kwargs)
                    scan.cutplot(show=True)
                    changed = True
            else:
                print(
                    "Specify the number of the cut you want to modify "
                    "as an argument after the command."
                )

        elif x == "ndslices":
            if len(split) < 2:
                print("Enter the requested number of dslices")
                ind = _cli_input(mode="alpha-integer")
            else:
                ind = split[1]
            scan.set_ndslices(int(ind))
            scan.cutplot(show=True)

            changed = True

        elif x in ["p", "pi"]:
            if x == "pi":
                deselected_tracks = scan.current_subset.apply_cuts(
                    scan._tracks, invert=True
                )
                scan.cutplot(show=True, tracks=deselected_tracks)
            else:
                scan.cutplot(show=True)

        elif x == "r":
            if len(split) < 2:
                print("Select the index of the cut to remove")
                ind = _cli_input(mode="integer")
            else:
                ind = split[1]
            print(f"Removing cut {int(ind)}")
            scan.current_subset.remove_cut(int(ind))
            scan.cutplot(show=True)

            changed = True

        elif x == "subset":
            if len(split) < 2:
                print(
                    "Select the index of the subset to switch to, or "
                    "enter 'new' to create a new subset."
                )
                ind = _cli_input(mode="alpha-integer")
            else:
                ind = split[1]

            if ind == "new":
                ind = len(scan._subsets)
                print(f"Creating a new subset, index {ind}")
                subset = Subset()
                scan.add_subset(subset)

            print(f"Selecting subset {ind}")
            scan.select_subset(int(ind))
            scan.cutplot(show=True)
            changed = True

        else:
            print(f"Invalid input: {x}")

    return changed
