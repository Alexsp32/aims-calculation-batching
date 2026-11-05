import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import ase.io
    import pathlib
    import matplotlib.pyplot as plt
    import ase.calculators.aims as aims_calc
    import ase.visualize.plot
    import uuid
    import datetime
    import re
    import shutil
    import numpy as np

    return aims_calc, ase, datetime, mo, pathlib, re, shutil, uuid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Structure tagger and batch FHI-AIMS calculation set-up
    This notebook sets up a default set of metadata added to an ase-compatible set of structures. If requested, tagged FHI-AIMS calculations are set up, ready for transfer and submission to a HPC cluster of choice.

    ## Structure input
    Please select a structure file below. Structures are visualised for reference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    file_upload = mo.ui.file(
        kind="area",
        filetypes=[".cif", ".xyz", ".traj"],
        multiple=False,
        label="Upload structure file here",
    )
    return (file_upload,)


@app.cell
def _(file_upload):
    file_upload
    return


@app.cell(hide_code=True)
def _(ase, file_upload, mo, pathlib):
    if pathlib.Path(mo.notebook_dir()).joinpath("tmp").is_dir() == False:
        pathlib.Path(mo.notebook_dir()).joinpath("tmp").mkdir()
    with open("tmp/input_file", "wb") as fi:
        fi.write(file_upload.contents())
    structures = ase.io.read("tmp/input_file", index=":")
    return (structures,)


@app.cell(hide_code=True)
def _(ase, mo, structures):
    extended_info = []
    for s in structures:
        info = {}
        info["Number of atoms"] = len(s)
        if type(s.calc) == ase.calculators.singlepoint.SinglePointCalculator:
            info.update(s.calc.results)
        info.update(s.info)
        extended_info.append(info)
    info_table = mo.ui.table(
        data = extended_info,
        pagination=True,
    )
    structure_to_show = mo.ui.slider(
        label="Select structure to show",
        start=1,
        stop=len(structures),
        step=1,
        value=1,
        include_input=True,
    )
    return info_table, structure_to_show


@app.cell(hide_code=True)
def _(info_table, mo, structure_to_show, structures):
    mo.md(r"""
    Loaded **{}** structures from input file. Check the visualisation below to confirm these are the structures you're interested in. 

    ### Structure metadata

    {}

    ---

    ### Structure visualisation

    {}
    """.format(len(structures), info_table, structure_to_show, structure_to_show.value))
    return


@app.cell(hide_code=True)
def _(ase, structure_to_show, structures):
    fig = ase.visualize.plot.plot_atoms(structures[structure_to_show.value-1], show_unit_cell=2)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Add structure metadata
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    structure_author = mo.ui.text(placeholder="Author name")
    structure_origin = mo.ui.text(placeholder="Where do the structures come from?")
    return structure_author, structure_origin


@app.cell(hide_code=True)
def _(mo):
    do_aims_calc = mo.ui.switch(value=False, label="Set up FHI-AIMS calculations?")
    return (do_aims_calc,)


@app.cell(hide_code=True)
def _(do_aims_calc, mo, structure_author, structure_origin):
    mo.md(r"""
    **Any of the information added here will be tagged onto each structure, as long as the same field doesn't already exist.**

    ### Author name
    Who created the structure files?

    {}

    ### Origin
    Where do the structures come from? Are they from simulation trajectories? DFT calculations? etc...

    {}

    ### UUID
    A unique ID automatically generated for each structure, which can be used to track the structure across databases and calculations. 

    ### DateAdded
    The date that the structure was added to the database. This is automatically generated. 

    ### LastUpdated
    The date that any structure information was last updated. This is automatically generated. 

    ### Perform DFT calculations?
    Check this box to set up a batch of FHI-AIMS calculations for these structures.

    {}

    """.format(
            structure_author,
            structure_origin,
            do_aims_calc
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    aims_placeholder = r"""{
        "xc": "srp 0.50",
        "relativistic": ["atomic_zora scalar"],
        "spin": "none",
    }
    """
    slurm_placeholder = r"""#!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --job-name="friction"
    #SBATCH --ntasks-per-node=128
    #SBATCH --cpus-per-task=1
    #SBATCH --time=8:00:00
    #SBATCH --output=%j.out
    #SBATCH --partition=highmem
    #SBATCH --qos=highmem

    module purge
    module load GCCcore/13.2.0 GCC/13.2.0 Python/3.11.5 OpenMPI/4.1.6
    module load FHI-aims/240507

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export MKL_DYNAMIC=FALSE
    ulimit -s unlimited

    echo "Running FHI-AIMS in directory:"
    pwd

    srun aims.x &> aims.out
    """
    aims_settings = mo.ui.code_editor(
        language="python",
        value=aims_placeholder,

    )
    slurm_settings = mo.ui.code_editor(
        language="bash",
        value=slurm_placeholder,
    )
    aims_x_path = mo.ui.text(placeholder="Path to FHI-AIMS executable")
    aims_species_dir = mo.ui.text(placeholder="Path to FHI-AIMS species directory")
    aims_calc_friction_enable_switch = mo.ui.switch(value=False, label="Apply friction to all H atoms?")
    aims_calc_basedir = mo.ui.text(value="calculations")
    return (
        aims_calc_basedir,
        aims_calc_friction_enable_switch,
        aims_settings,
        aims_species_dir,
        aims_x_path,
        slurm_settings,
    )


@app.cell(hide_code=True)
def _(
    aims_calc_basedir,
    aims_calc_friction_enable_switch,
    aims_settings,
    aims_species_dir,
    aims_x_path,
    do_aims_calc,
    mo,
    slurm_settings,
):
    if do_aims_calc.value:
        md = mo.md(
            """
            ### FHI-AIMS settings
            Please input the settings you'd like to use for your FHI-AIMS calculations. These should be input as a python dictionary, with the same keys as those used in the `ase.calculators.aims.Aims` calculator. For example, the default settings are shown below. 

            {}

            ### FHI-AIMS command
            Please input the whole command needed to make your calculation run on the cluster in question. This will be used to build a job array submission script for all calculations.

            {}

            ### FHI-AIMS species directory
            Please input the path to your FHI-AIMS species directory. This is required to set up the calculation input files correctly. 

            {}

            ### Calculation base directory
            Folder name under which the calculations will be stored. 

            {}

            ### Submit script
            Please copy the SLURM submit script with your cluster specific settings here. This script will be used as the basis for the job array submission file. Array-specific settings will be appended to the end of the `#SBATCH` block. 

            {}

            ### Specific parameters for friction calculations

            {}
            """.format(
                aims_settings,
                aims_x_path,
                aims_species_dir,
                aims_calc_basedir,
                slurm_settings,
                aims_calc_friction_enable_switch,
            )
        )
    else:
        md = mo.md("")
    md
    return


@app.cell
def _(mo):
    new_structure_filename = mo.ui.text(value="structures.xyz", label="Where should the structures with added metadata be saved?")
    run_button = mo.ui.run_button(label="Save structures and set up calculations")
    return new_structure_filename, run_button


@app.cell
def _(mo, new_structure_filename, run_button):
    mo.md(r"""
    Make sure the settings you've input above are correct, then click the button below to save the structures with the added metadata, and set up the FHI-AIMS calculation directories (if requested).

    {}

    {}
    """.format(new_structure_filename, run_button)
    )
    return


@app.cell
def _(datetime, structures):
    datetime.datetime.now()
    structures[0].info.get("UUID")
    return


@app.cell
def _(pathlib):
    test_path = pathlib.Path("../structures/gen7_dataset_no_dimer.extxyz")
    other_base_path = pathlib.Path("../").resolve()
    test_path = test_path.resolve()
    test_path.relative_to(other_base_path).as_posix()
    return


@app.cell
def _(
    aims_calc,
    aims_calc_basedir,
    aims_calc_friction_enable_switch,
    aims_settings,
    aims_species_dir,
    aims_x_path,
    ase,
    datetime,
    do_aims_calc,
    mo,
    new_structure_filename,
    pathlib,
    re,
    run_button,
    shutil,
    slurm_settings,
    structure_author,
    structure_origin,
    structures,
    uuid,
):
    if run_button.value:
        print("Successfully started")
        for structure in structures:
            if structure_author.value != "":
                if "author" not in structure.info.keys():
                    structure.info["author"] = structure_author.value
            if structure_origin.value != "":
                if "type" not in structure.info.keys():
                    structure.info["type"] = structure_origin.value
            if structure.info.get("UUID") is None:
                structure.info["UUID"] = str(uuid.uuid4())
            if structure.info.get("DateAdded") is None:
                # Return date in format YYYY-MM-DD HH:MM
                structure.info["DateAdded"]=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            structure.info["LastUpdated"]=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        structure_output_path = pathlib.Path(new_structure_filename.value)
        print("Successfully wrote structure metadata. ")
        print("Saving structures to {}".format(structure_output_path))
        ase.io.write(structure_output_path, structures)
        if do_aims_calc.value:
            print("Setting up FHI-AIMS calculations")
            aims_settings_dict = eval(aims_settings.value)
            aims_profile = aims_calc.AimsProfile(
                command=aims_x_path.value,
                default_species_directory=aims_species_dir.value,
            )
            def make_calculator(directory, parameters):
                return aims_calc.Aims(
                    profile=aims_profile,
                    directory=directory,
                    **parameters,
                )
            base_path = pathlib.Path(mo.notebook_dir()).joinpath("aims-calculations", aims_calc_basedir.value)
            if not base_path.exists():
                base_path.mkdir(parents=True)
            else:
                raise ValueError("Calculation directory already exists. ")
            calculation_directories = []
            for i, structure in enumerate(structures):
                calculation_uuid = str(uuid.uuid4())
                calc_dir = base_path.joinpath(calculation_uuid)
                # Make calculation directory
                calc_dir.mkdir()
                # Add to full list so I can make a job array. 
                calculation_directories.append(calc_dir.relative_to(base_path))
                # Save a copy of the structure with metadata in the calculation dir to associatete it with the calculation. 
                ase.io.write(calc_dir.joinpath("structure.xyz"), structure)
                # Generate AIMS calculation
                structure.calc = make_calculator(calc_dir, aims_settings_dict)
                structure.calc.write_inputfiles(structure, ["energy"])
                # Save Slurm submit file. 
                # with open(calc_dir.joinpath("submit.sh"), "w") as submit_file:
                #     submit_file.write(slurm_settings.value)
                # Add to batch submission script. 
                # big_submit_script.writelines(
                #     [
                #         "cd $BASE_DIR/{}\n".format(calc_dir.stem),
                #         "sbatch submit.sh\n",
                #     ],
                # )
                if aims_calc_friction_enable_switch.value:
                    # This is a workaround to enable friction calculations since the ASE calculator doesn't have a way of adding tags to individual atoms. 
                    geometry_file_obj = open(structure.calc.directory.joinpath("geometry.in"), "r")
                    geometry_file = geometry_file_obj.readlines()
                    geometry_file.append("\n")
                    geometry_file_obj.close()
                    # Overwrite geometry file with added friction flags. 
                    with open(structure.calc.directory.joinpath("geometry.in"), "w") as file:
                        for line_number,line_string in enumerate(geometry_file):
                            if re.search(r"atom.*H\n", line_string) and geometry_file[line_number+1]!="\tcalculate_friction .true.\n":
                                geometry_file.insert(line_number+1, "\tcalculate_friction .true.\n") # Add calculate_friction tag
                        file.writelines(geometry_file)
            # Update array specs in the submit script. 
            # Find last line of the #SBATCH block and append array spec. 
            # Bash script to submit all individual SLURM jobs:
            big_submit_script = open(base_path.joinpath("submit_all.sh"), "w")
            slurm_base = slurm_settings.value.split("\n") # Split by lines for easier modification. 
            slurm_base = [line + "\n" for line in slurm_base] # Add back newlines
            # Find last #SBATCH line
            last_sbatch_line = -["#SBATCH" in line for line in slurm_base][::-1].index(True)
            big_submit_script.writelines(
                slurm_base[:last_sbatch_line]
            )
            big_submit_script.write("#SBATCH --array=0-{}\n".format(len(calculation_directories)-1))
            big_submit_script.writelines(
                slurm_base[last_sbatch_line+1:]
            )
            # Then define the list of folders and cd based on SLURM_ARRAY_TASK_ID
            big_submit_script.write("calculation_folders=(\n") # Start array
            big_submit_script.writelines([dir.as_posix() + " \\\n" for dir in calculation_directories]) # array contents
            big_submit_script.write(")\n") # End array
            big_submit_script.write('cd ${calculation_folders[$SLURM_ARRAY_TASK_ID]}\n echo "Running calculation in $(pwd)"\n')
            big_submit_script.write(aims_x_path.value + "\n")
            # NTFY push
            big_submit_script.write(
                'curl -d "{}: SLURM job "$SLURM_JOBID": "$(sacct -j $SLURM_JOBID -o State | tail -n1) ntfy.sh/zcYDEh20bV2t3hMe'.format(aims_calc_basedir.value)
            )
            big_submit_script.close()
            # Zip the whole calculation folder for download. Needs to be compatible with saving for WebAssembly. 
            shutil.make_archive(aims_calc_basedir.value, 'zip', str(base_path))
            dl_calcs = mo.download(
                open("{}.zip".format(aims_calc_basedir.value), "rb"), 
                label="Download AIMS inputs",
                filename="{}.zip".format(aims_calc_basedir.value),
                mimetype="application/zip",
            )
        dl_str = mo.download(
            open(structure_output_path, "rb"), 
            label="Download structures with metadata",
            filename=structure_output_path.name,
        )
    else:
        dl_str = mo.download(None,disabled=True, label="Download structures with metadata")
        dl_calcs = mo.download(None,disabled=True, label="Download AIMS inputs")
    return dl_calcs, dl_str


@app.cell
def _(dl_calcs, dl_str, mo):
    mo.hstack(
        [
            dl_str,
            dl_calcs,
        ],
        justify="start"
    )
    return


if __name__ == "__main__":
    app.run()
