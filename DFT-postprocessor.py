import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="DFT batch postprocessor",
    auto_download=["ipynb"],
)


@app.cell
def _():
    import marimo as mo
    import ase.io
    import ase
    import ase.visualize.plot
    import numpy as np
    import datetime
    import pathlib
    import dfttoolkit.output
    import dfttoolkit.friction
    import shutil
    import json
    import tempfile

    return ase, datetime, dfttoolkit, mo, np, pathlib, shutil, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FHI-AIMS Calculation $\rightarrow$ XYZ file postprocessor

    This is the counterpart to the calculation set-up notebook. It reads out the DFT results from a zip archive and adds them to a structure XYZ file, updating replaced values if the calculated results are newer.
    """)
    return


@app.cell
def _(mo):
    dft_zip_archive = mo.ui.file(
        kind="area",
        filetypes=[".zip"],
        multiple=False,
        label="Drop AIMS results here",
        max_size= 10000000000 # 10GB should be enough (please)
    )
    return (dft_zip_archive,)


@app.cell
def _(mo):
    structure_db_file = mo.ui.file(
        kind="area",
        filetypes=[".xyz"],
        multiple=False,
        label="Drop structure file to update here",
    )
    return (structure_db_file,)


@app.cell
def _(mo):
    dft_zip_path = mo.ui.text(value = "")
    return (dft_zip_path,)


@app.cell
def _(mo):
    friction_method_selector = mo.ui.dropdown(
        options = [
            "positive_excitation",
            "negative_excitation",
        ],
        label = "Select the ODF implementation used in your calculations",
        value = "negative_excitation",
    )
    return (friction_method_selector,)


@app.cell
def _(
    dft_zip_archive,
    dft_zip_path,
    friction_method_selector,
    mo,
    structure_db_file,
):
    mo.md(r"""
    ### Calculation results

    {}

    or give a file path to extracted results here.

    {}

    ### Structure file to update
    This file should already contain all the structures that a calculation was started for. This is checked by reading the UUID field. 

    {}

    ### ODF implementation

    In Oct 2026, the ODF implementation in FHI-AIMS was changed to include negative excitations, as they were found to alter EFTs significantly, leading to more correct values. 

    Please select which friction implementation was used in your output files: {}

    The `"ODF_method"` key specifies which state of implementation was used in files saved by this notebook.: 

    | `"ODF_method"` value             | Explanation                                                                                                                    |
    | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
    | `positive_excitation`            | Implementation before discovering the negative excitation issue in October 2026.                                               |
    | `positive_excitation_multiplied` | Pre- October 2026 values multiplied with a corrective factor of 2 in post. This should be equivalent to `negative_excitation`. |
    | `negative_excitation`            | Corrected implementation in the `electron_phonon_symmetry_fixes` branch (or later release which merges this branch).           |

    - The ODF EFT is contained as a JSON-readable row-major array in `"ODF_EFT"`. 
    	- Only the friction-active blocks of the ODF EFT are given, with each 3x3 block corresponding to the order of H atoms in the structure. 
    - `"ODF_EFT_units"` specifies EFT units, which should be ps^-1 by default. 

    """.format(dft_zip_archive, dft_zip_path, structure_db_file, friction_method_selector))
    return


@app.cell
def _():
    """
    Accessor and save layout for energies. 

    I use a custom structure here to avoid ambiguity even if it makes plotting a bit slower since the typical ASE calculator interface doesn't work. 
    """
    def read_energy(dfttoolkit_output, structure):
        if "DFT_energy" not in structure.info.keys():
            structure.info["DFT_energy"] = dfttoolkit_output.get_final_energy() # I assume this is in eV?
            structure.info["DFT_energy_unit"] = "eV"
            return True
        else:
            return False

    return (read_energy,)


@app.cell
def _():
    """
    Accessor and save layout for forces. 
    """
    def read_forces(dfttoolkit_output, structure):
        if "DFT_forces" not in structure.info.keys():
            try:
                forces = dfttoolkit_output.get_forces()
            except:
                return False
            structure.arrays["DFT_forces"] = forces # I assume this is in eV/Å?
            structure.info["DFT_forces_unit"] = "eV/Å"
            return True
        else:
            return False

    return (read_forces,)


@app.cell
def _(dfttoolkit, friction_method_selector, np):
    """
    Accessor and save layout for EFTs. 
    """
    def read_EFT(calculation_path, structure):
        if "ODF_EFT" not in structure.info.keys() and calculation_path.joinpath("friction_tensor.out").is_file():
            EFT = dfttoolkit.friction.FrictionTensor(calculation_path).friction_tensor
            EFT_H = EFT[
                np.ix_( # Slice across all H indices times 3 since the tensor is 3x3 blocks for every atom
                    np.repeat(structure.symbols, 3) == 'H',
                    np.repeat(structure.symbols, 3) == 'H',
                )
            ]
            # Check if old or new friction. Multiply old friction by 2. 
            if friction_method_selector.value == "positive_excitation":
                EFT_H *= 2
                structure.info["ODF_EFT_implementation"] = "positive_excitation_multiplied"
            else:
                structure.info["ODF_EFT_implementation"] = friction_method_selector.value
            structure.info["ODF_EFT"] = EFT_H
            structure.info["ODF_EFT_unit"] = "ps^-1"
            return True
        else:
            return False

    return (read_EFT,)


@app.cell
def _(calc_folder, friction_method_selector, np):
    """
    Accessor and save layout for friction spectra. 
    (only ascii formatted so far. Still need to add HDF5 support) 
    """
    def read_friction_spectrum(calculation_path, structure):
        key_not_written = "ODF_friction_spectrum" not in structure.info.keys()
        spectrum_available = calculation_path.joinpath("friction_tensor_grid.out").is_file()
        if key_not_written and spectrum_available:
            with open(calc_folder.joinpath("friction_tensor_grid.out"), "r") as f:
                lines = f.readlines()
                # Find line with "Number of bins"
                for line in lines:
                    if "Number of Bins" in line:
                        num_bins = int(line.split()[-1])
                        break
                # Find all lines with "Friction component for"
                spectrum_starts = [i for i, line in enumerate(lines) if "Friction component for" in line]
                # Find total EFT size from components of last line
                eft_size = (int(lines[spectrum_starts[-1]].split()[-1]), int(lines[spectrum_starts[-1]].split()[-2]))
                # Initialize friction spectrum array with complex numbers
                friction_spectrum = np.zeros((eft_size[0], eft_size[1], num_bins), dtype=complex)
                # Parse friction spectrum data
                for start in spectrum_starts:
                    component = lines[start].split()[3:]  # Get component name
                    excitation_energies = []
                    i, j = map(int, component)  # Get indices from component name
                    for k in range(num_bins):
                        line = lines[start + 1 + k].split()
                        friction_spectrum[i-1, j-1, k] = complex(real = float(line[1]), imag = float(line[2]))  # Get friction value
                        excitation_energies.append(float(line[0]))  # Get excitation energy
            # Save outputs to XYZ structure in the info fields. 
            structure.info["ODF_friction_spectrum"] = friction_spectrum
            structure.info["ODF_excitation_energies"] = excitation_energies
            structure.info["ODF_excitation_energies_unit"] = "eV"
            structure.info["ODF_friction_spectrum_unit"] = "ps^-1"
            structure.info["ODF_friction_spectrum_implementation"] = friction_method_selector.value
            return True
        else:
            return False

    return (read_friction_spectrum,)


@app.cell
def _(
    ase,
    datetime,
    dfttoolkit,
    read_EFT,
    read_energy,
    read_forces,
    read_friction_spectrum,
):
    """
    This function processes one calculation by checking if the structure is in the database and adding all available results to the database if they are not already present. It returns True if the structure was found in the database and False if it was not.
    """
    def extract_calculation_information(calculation_path, ase_atoms_list, meta_info):
        # Check if calculation was started - If not we don't need to read anything
        potential_output_files = [pt for pt in calculation_path.glob("aims*.out")] # In case you have a pattern like aims_SLURM_JOBID.out
        if len(potential_output_files)==0:
            print("Calculation never finished: {}".format(calculation_path))
            meta_info["num_scf_cycles"].append(None)
            meta_info["time_per_scf"].append(None)
            meta_info["was_successful"].append(False)
            return False
        output_file = potential_output_files[-1] # Pick the last file for no particular reason. 
        dt_output = dfttoolkit.output.AimsOutput(aims_out = output_file)
        # Calculation exists, so check if it is in db and merge
        initial_structure = ase.io.read(calculation_path.joinpath("structure.xyz"))
        # unless it didn't finish properly. 
        try: 
            exited_normally = dt_output.check_exit_normal()
        except:
            exited_normally = False
        if not exited_normally:
            print("Calculation {} started running, but failed.".format(calculation_path))
            meta_info["num_scf_cycles"].append(dt_output.get_n_scf_iters())
            meta_info["time_per_scf"].append(dt_output.get_time_per_scf())
            meta_info["was_successful"].append(False)
            return False
        db_uuids = [db_structure.info["UUID"] for db_structure in ase_atoms_list]
        structure_uuid = initial_structure.info["UUID"]
        if structure_uuid in db_uuids:
            # Add meta
            meta_info["num_scf_cycles"].append(dt_output.get_n_scf_iters())
            meta_info["time_per_scf"].append(dt_output.get_time_per_scf())
            # Update information from DFT calculations. Save structure prototypes are defined above.  
            db_index = db_uuids.index(structure_uuid)
            forces = read_forces(dt_output, ase_atoms_list[db_index])
            energy = read_energy(dt_output, ase_atoms_list[db_index])
            eft = read_EFT(calculation_path, ase_atoms_list[db_index])
            spectrum = read_friction_spectrum(calculation_path, ase_atoms_list[db_index])
            updated = forces or energy or eft or spectrum
            # Update LastUpdated if we added new information. 
            if updated:
                ase_atoms_list[db_index].info["LastUpdated"] = datetime.datetime.strftime(
                    datetime.datetime.now(),
                    "%Y-%m-%d %H:%M",
                )
            return True
        else:
            print("Found structure with no associated db record: {}".format(structure_uuid))
            return False

    return (extract_calculation_information,)


@app.cell
def _(mo):
    buton = mo.ui.run_button(label="Post-process calculations")
    return (buton,)


@app.cell
def _(pathlib, tempfile):
    with tempfile.TemporaryDirectory() as f:
        print(pathlib.Path(f))
    return


@app.cell
def _():
    max([1,2,3])
    return


@app.cell
def _(
    ase,
    buton,
    dft_zip_archive,
    dft_zip_path,
    extract_calculation_information,
    mo,
    np,
    pathlib,
    shutil,
    structure_db_file,
    tempfile,
):
    if buton.value:
        out = ""
        calculation_metadata = { # Show this information after processing to get an idea of calculation times. 
            "was_successful" : [],
            "num_scf_cycles" : [],
            "time_per_scf" : [],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            out += "Extracted results to temporary directory: {}\n\n".format(tmp_path)
            with open(tmp_path.joinpath("structures.xyz"), "wb") as fi:
                fi.write(structure_db_file.contents())
            structure_db = ase.io.read(tmp_path.joinpath("structures.xyz"), index=":")
            if not dft_zip_archive and not dft_zip_path.value:
                error_msg = "No calculation results given. Please provide a zip archive or path to extracted results."
                raise(ValueError(error_msg))
            elif dft_zip_path.value:
                zip_path = pathlib.Path(dft_zip_path.value)
            elif dft_zip_archive.value:
                # Unzip calculation results to the temp directory
                with open(tmp_path.joinpath(dft_zip_archive.name()), "wb") as archive_file:
                    archive_file.write(dft_zip_archive.contents())
                shutil.unpack_archive(tmp_path.joinpath(dft_zip_archive.name()), tmp_path)
                zip_path = tmp_path
            # Now files are available, so run postprocessing. 
            potential_folders = [path for path in zip_path.glob("*") if path.is_dir()]
            for folder in potential_folders:
                extract_calculation_information(folder, structure_db, calculation_metadata)
            # Output updated structures and give download button. 
            ase.io.write(tmp_path.joinpath(structure_db_file.name()), structure_db)
            structures_dl = mo.download(
                open(tmp_path.joinpath(structure_db_file.name()), "rb"),
                label="Download updated structures"
            )
            out += "{}\n\n".format(structures_dl)
            # Output overview
            extended_info = []
            for s in structure_db:
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
            out += "### Updated structure information\n\n{}\n\n".format(info_table)
            # Generate metadata info
            out += """
                #### Calculation statistics

                Postprocessed calculations: {}

                Average number of SCF steps: {}
                Average time per SCF steps: {}

                Maximum number of SCF steps: {}
                Maximum time for a single SCF step: {}
            """.format(
                len(calculation_metadata["was_successful"]),
                np.mean([n for n in calculation_metadata["num_scf_cycles"] if n is not None]),
                np.mean([np.mean(time) for time in calculation_metadata["time_per_scf"] if time is not None]),
                np.max([n for n in calculation_metadata["num_scf_cycles"] if n is not None]),
                np.max(np.concatenate([time.flatten() for time in calculation_metadata["time_per_scf"] if time is not None])),
            )
    else:
        out = "Postprocessing results will be shown here after you click the button."
    return (out,)


@app.cell(hide_code=True)
def _(buton, mo, out):
    mo.md(r"""
    ### Postprocessing

    {}

    {}

    """.format(buton, out))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
