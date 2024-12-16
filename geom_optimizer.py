# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:13:29 2024

@author: matar
"""

from contextlib import redirect_stdout
import time
import sys
import os

import numpy as np
import pandas as pd


# defining types for readability
Vector = list[float]
Matrix = np.array([int, int])

# set precision of the printing
np.set_printoptions(
    precision=3, suppress=True, threshold=np.inf, linewidth=7000
)

# pylint: disable=line-too-long
# pylint: disable=too-many-lines
# pylint: disable=redefined-outer-name
# pylint: disable=no-member
# pylint: disable=invalid-name

################################################################################
#                                                                              #
#                        Definition of the force field                         #
#                                                                              #
################################################################################

# We start by defining a set of lists that have the internal coordinate type,
# force constant, and equilibrium distances associate with each type of internal
# coordinates. This way, it will be possible to know all information from an
# internal coordinate defined later by indexing an identifiyng integer as
# prop[identifier_int]

COORDINATE_TYPE = np.array(["C-C", "C-H", "H-C-H", "H-C-C", "C-C-C", "X-C-C-X"])
F_CONST = np.array([300, 350, 35, 35, 60, 0.3])
EQ_DIST = np.array([1.53, 1.11, 109.50, 109.50, 109.50, 3])

# We define here the VdW interaction constants
WDW_ATOM_LABELS = np.array(["H", "C"])
VDW_EPSILON_i = np.array([0.03, 0.07])
VDW_SIGMA_i = np.array([1.2, 1.75])
N_VDW_PARAMETERS = len(VDW_SIGMA_i)


# The choice to define VDW_SIGMA_ij and VDW_EPSILON_ij this way was done thinking
# that it would make the handling of adding new atom types to the field easily
# by simply appending the new values to the previous lists
VDW_SIGMA_ij = np.zeros((N_VDW_PARAMETERS, N_VDW_PARAMETERS))
VDW_EPSILON_ij = np.zeros((N_VDW_PARAMETERS, N_VDW_PARAMETERS))


# We fill the VDW_SIGMA_ij and VDW_EPSILON_ij arrays
for i in range(N_VDW_PARAMETERS):
    for j in range(i, N_VDW_PARAMETERS):
        VDW_SIGMA_ij[i][j] = VDW_SIGMA_ij[j][i] = (
            2 * (VDW_SIGMA_i[i] * VDW_SIGMA_i[j]) ** 0.5
        )

for i in range(N_VDW_PARAMETERS):
    for j in range(i, N_VDW_PARAMETERS):
        VDW_EPSILON_ij[i][j] = VDW_EPSILON_ij[j][i] = (
            VDW_EPSILON_i[i] * VDW_EPSILON_i[j]
        ) ** 0.5

# we define the Lennard-Jones A and B parameters
LJ_A_ij = np.zeros((N_VDW_PARAMETERS, N_VDW_PARAMETERS))
LJ_B_ij = np.zeros((N_VDW_PARAMETERS, N_VDW_PARAMETERS))

for i in range(N_VDW_PARAMETERS):
    for j in range(i, N_VDW_PARAMETERS):
        LJ_A_ij[i][j] = LJ_A_ij[j][i] = (
            4 * (VDW_SIGMA_ij[i][j]) ** 12 * VDW_EPSILON_ij[i][j]
        )

for i in range(N_VDW_PARAMETERS):
    for j in range(i, N_VDW_PARAMETERS):
        LJ_B_ij[i][j] = LJ_B_ij[j][i] = (
            4 * (VDW_SIGMA_ij[i][j]) ** 6 * VDW_EPSILON_ij[i][j]
        )

################################################################################
#                                                                              #
#                                File handling                                 #
#                                                                              #
################################################################################

# in this section we will define some functions to transform raw mol2 file to useful data
# some choices were made knowing that these are not the most efficient ways to write the
# functions, but for the sake of clarity were written in what I thought was the most
# understandable way.


def read_mol2_file(mol2_file: str) -> list[str]:
    """Read a mol2 file and returns the content as a list."""

    with open(mol2_file, "r", encoding="utf-8") as file:
        content_list = file.readlines()
    return content_list


def identify_sections(content_list: list) -> Vector:
    """Identify the start of the sections of the mol2 file.

    Useful to avoid iterations over the content list.
    """

    markers = []
    for line, content in enumerate(content_list):
        if "@<" in content:
            markers.append(line)
    return markers


def get_atoms_section(content_list: list[str], markers: list[int]) -> Vector:
    """Analyze the labels in the mol2 file to determine the section that
    contains the atom information"""

    starting_atoms = end_of_atoms = False

    for marker in markers:
        if starting_atoms:
            end_of_atoms = marker
            break

        if (
            content_list[marker].strip().capitalize()
            == "@<TRIPOS>ATOM".capitalize()
        ):
            # .capitalize() to avoid probles with lowercase or uppercase
            starting_atoms = marker + 1

    if not end_of_atoms:
        end_of_atoms = len(content_list)

    start_end_atoms = [starting_atoms, end_of_atoms]

    return start_end_atoms


def get_coordinates(
    content_list: list[str], start_end_atoms: list[int]
) -> Matrix:
    """Using the marker list, reads the content list and extracts the number
    of atoms, atom labels, coordinates as a matrix and coordinates as an array.

    The starting and ending of atoms was defined this way in case the sections
    were not in order. Special precautions could be taken to avoid problems
    with blank lines
    """

    starting_atoms, end_of_atoms = start_end_atoms

    atoms = content_list[starting_atoms:end_of_atoms]

    atom_coord_2d = np.array(
        [atom.strip().split()[2:5] for atom in atoms], dtype=np.float64
    )

    return atom_coord_2d


def get_labels(
    content_list: list[str], start_end_atoms: list[int]
) -> list[str]:
    """Using the marker list, reads the content list and extracts the atom
    labels
    """

    starting_atoms, end_of_atoms = start_end_atoms

    atoms = content_list[starting_atoms:end_of_atoms]

    ATOM_LABELS = [atom.strip().split()[1] for atom in atoms]

    return ATOM_LABELS


def read_unconventional_mol2(mol2_file: str) -> list:
    """Read a mol2 file that doesn't have differenciated sections with @<_>."""

    ATOM_LABELS = []
    atom_coord_2d = []

    with open(mol2_file, "r", encoding="utf-8") as file:
        content_list = file.readlines()

    for line in content_list:
        if len(line.strip().split()) == 16:
            ATOM_LABELS.append(line.strip().split()[3])
            atom_coord_2d.append(line.strip().split()[0:3])

    number_of_atoms = len(ATOM_LABELS)
    atom_coord_2d = np.array(atom_coord_2d, dtype=float)
    CONNECTIVITY_MATRIX = np.zeros([number_of_atoms, number_of_atoms])

    for line in content_list:
        if len(line.strip().split()) == 7:
            i, j = line.strip().split()[0:2]
            CONNECTIVITY_MATRIX[int(i) - 1][int(j) - 1] = CONNECTIVITY_MATRIX[
                int(j) - 1
            ][int(i) - 1] = 1

    return [number_of_atoms, ATOM_LABELS, atom_coord_2d, CONNECTIVITY_MATRIX]


def extend_labels() -> list[str]:
    """Extend the labels to add an identifiying integer."""

    counter_labels = list(set(ATOM_LABELS))
    counter = np.zeros(len(counter_labels))

    extended_labels = []

    for label in ATOM_LABELS:
        counter[counter_labels.index(label)] += 1
        extended_labels.append(
            label + str(int(counter[counter_labels.index(label)]))
        )

    return extended_labels


def get_CONNECTIVITY_MATRIX(
    content_list: list[str], markers: list[int]
) -> Matrix:
    """Using the marker list and the number of atoms, reads the content list and extracts
    the connectivity matrix. It will be a N_ATOMS by N_ATOMS diagonally symmetric matrix
    that will have 0 or 1 depending on if two atoms are connected or not. It will be
    useful later to define the internal coordinates."""

    starting_connectivity = end_of_connectivity = False
    for marker in markers:
        if starting_connectivity:
            end_of_connectivity = marker
            break

        if (
            content_list[marker].strip().capitalize()
            == "@<TRIPOS>BOND".capitalize()
        ):
            starting_connectivity = marker + 1

    if not end_of_connectivity:
        end_of_connectivity = len(content_list)

    bonds = content_list[starting_connectivity:end_of_connectivity]
    bonds = [np.array(bond.strip().split(), dtype=int) for bond in bonds]

    CONNECTIVITY_MATRIX = np.zeros([N_ATOMS, N_ATOMS])

    for bond in bonds:
        CONNECTIVITY_MATRIX[bond[1] - 1][bond[2] - 1] = CONNECTIVITY_MATRIX[
            bond[2] - 1
        ][bond[1] - 1] = 1

    return CONNECTIVITY_MATRIX


################################################################################
#                                                                              #
#                    Defining bonds, angles and dihedrals                      #
#                                                                              #
################################################################################


def get_BOND_LIST() -> Matrix:
    """Builds the bond list using the connectivity matrix"""

    N_ATOMS = len(CONNECTIVITY_MATRIX)

    BOND_LIST = []

    for i in range(N_ATOMS):
        for j in range(i, N_ATOMS):
            if CONNECTIVITY_MATRIX[i][j] == 1:
                BOND_LIST.append([i, j])

    return BOND_LIST


def build_angles2(BOND_LIST) -> list:
    """Builds the angle list by adding an atom to the end of the bonds
    forwards and backwards. If the angle is new it is stored. If not,
    discarded"""

    ANGLE_LIST = []
    angle_raw_list = []  # placeholder list for bonds that can be repeated

    for _ in range(2):  # we iterate twice, once forward and once backwards
        BOND_LIST = [bond[::-1] for bond in BOND_LIST]  # reverse bond indexes
        for bond in BOND_LIST:
            for atom in range(N_ATOMS):
                if (
                    CONNECTIVITY_MATRIX[bond[-1]][atom] == 1
                    and atom not in bond
                ):  # Check connectivity
                    angle_raw_list.append([bond[0], bond[1], atom])

    for angle in angle_raw_list:
        # check uniqueness (avoid repetition of the same angle indexes flipped)
        if angle not in ANGLE_LIST and angle[::-1] not in ANGLE_LIST:
            ANGLE_LIST.append(angle)

    ANGLE_LIST.sort(key=lambda x: x[0])  # sort the list by the first index

    return ANGLE_LIST


def build_dihedrals2(ANGLE_LIST) -> list:
    """Builds the dihedral list by adding an atom to the end of the angles
    forwards and backwards. If the dihedral is new it is stored. If not,
    discarded"""

    dihedral_raw_list = []
    DIHEDRAL_LIST = []

    for _ in range(2):  # same as before, two iterations
        ANGLE_LIST = [
            angle[::-1] for angle in ANGLE_LIST
        ]  # reverse angle indexes
        for angle in ANGLE_LIST:
            for atom in range(N_ATOMS):
                if (
                    CONNECTIVITY_MATRIX[angle[-1]][atom] == 1
                    and atom not in angle
                ):  # Check connectivity
                    dihedral_raw_list.append(
                        [angle[0], angle[1], angle[2], atom]
                    )

    for dihedral in dihedral_raw_list:
        if (
            dihedral not in DIHEDRAL_LIST
            and dihedral[::-1] not in DIHEDRAL_LIST
        ):  # check uniqueness
            DIHEDRAL_LIST.append(dihedral)

    DIHEDRAL_LIST.sort(key=lambda x: x[0])  # sort the list by the first index

    return DIHEDRAL_LIST


def calculate_r_vector_matrix(atom_coord_2d: Matrix) -> Matrix:
    """Calculates all interatomic vectors. Will be used to calculate
    distances, angles and dihedrals."""

    N_ATOMS = len(atom_coord_2d)
    r_vector_matrix = []

    for i in range(N_ATOMS):
        append_list = []
        for j in range(N_ATOMS):
            append_list.append(-(atom_coord_2d[j] - atom_coord_2d[i]))
        r_vector_matrix.append(append_list)

    return r_vector_matrix


def calculate_dist_mat(r_vector_matrix: Matrix) -> Matrix:
    """Calculates all distances between atoms. Will be used to calculate
    bond energies and VdW energies."""

    N_ATOMS = len(r_vector_matrix)
    dist_mat = np.zeros([N_ATOMS, N_ATOMS])

    for i in range(N_ATOMS):
        for j in range(i, N_ATOMS):
            dist_mat[i][j] = dist_mat[j][i] = np.linalg.norm(
                r_vector_matrix[i][j]
            )

    return np.array(dist_mat)


def calculate_angles(
    r_vector_matrix: Matrix, dist_mat: Matrix, radians=False
) -> Vector:
    """Calculates the angles using
    theta = arccos((r_ba · r_bc)/(||r_ba||*||r_bc||)) * 180/pi"""

    angle_values = []

    if radians:
        conversion_factor = 1
    else:
        conversion_factor = 180 / np.pi

    for angle in ANGLE_LIST:
        a, b, c = angle
        angle_values.append(
            float(
                np.arccos(
                    (r_vector_matrix[b][a] @ r_vector_matrix[b][c])
                    / (dist_mat[a][b] * dist_mat[b][c])
                )
                * conversion_factor
            )
        )

    return np.array(angle_values)


def calculate_dihedrals(r_vector_matrix: Matrix, radians=False) -> Vector:
    """Calculates the dihedral angles with the atan(sin_phi, cos_phi) formula"""

    dih_val = []

    if radians:
        conversion_factor = 1
    else:
        conversion_factor = 180 / np.pi

    for dihedral in DIHEDRAL_LIST:
        a, b, c, d = dihedral
        t = np.cross(r_vector_matrix[a][b], r_vector_matrix[b][c])
        u = np.cross(r_vector_matrix[b][c], r_vector_matrix[c][d])
        v = np.cross(t, u)

        cos_phi = t @ u / (np.linalg.norm(t) * np.linalg.norm(u))
        sin_phi = (r_vector_matrix[b][c] @ v) / (
            np.linalg.norm(t)
            * np.linalg.norm(u)
            * np.linalg.norm(r_vector_matrix[b][c])
        )

        phi = np.arctan2(sin_phi, cos_phi) * conversion_factor

        dih_val.append(phi)

    return dih_val


def identify_internals() -> list[list[int], list[int]]:
    """This function identifies all the internal coordinate types, generating
    two lists with integers that correspond to the index for the equilibrium
    constants and values"""

    BOND_TYPES = []
    ANGLE_TYPES = []
    for bond in BOND_LIST:
        bond_type = list(COORDINATE_TYPE).index(
            "%s-%s" % (ATOM_LABELS[bond[0]], ATOM_LABELS[bond[1]])
        )
        BOND_TYPES.append(bond_type)

    for angle in ANGLE_LIST:
        angle_str = "%s-%s-%s" % (
            ATOM_LABELS[angle[0]],
            ATOM_LABELS[angle[1]],
            ATOM_LABELS[angle[2]],
        )

        if angle_str in COORDINATE_TYPE:
            angle_type = list(COORDINATE_TYPE).index(angle_str)
            ANGLE_TYPES.append(angle_type)

        else:
            # we switch the order
            angle_str = "%s-%s-%s" % (
                ATOM_LABELS[angle[2]],
                ATOM_LABELS[angle[1]],
                ATOM_LABELS[angle[0]],
            )
            if angle_str in COORDINATE_TYPE:
                angle_type = list(COORDINATE_TYPE).index(angle_str)
                ANGLE_TYPES.append(angle_type)

    return BOND_TYPES, ANGLE_TYPES


def calculate_internal_values(atom_coord_2d: Matrix) -> Vector:
    """Calculates the value of each of the internal coordinates from the
    cartesian coordinates
    """

    r_vector_matrix = calculate_r_vector_matrix(atom_coord_2d)
    dist_mat = calculate_dist_mat(r_vector_matrix)

    bonds = np.array([dist_mat[i][j] for i, j in BOND_LIST])

    angles = calculate_angles(r_vector_matrix, dist_mat, radians=True)
    dihedrals = -np.array(
        calculate_dihedrals(r_vector_matrix, radians=True)
    )  # ???

    bonds_and_angles = np.append(bonds, angles)
    internals = np.append(bonds_and_angles, dihedrals)

    return internals


################################################################################
#                                                                              #
#                         Calculation of the energy                            #
#                                                                              #
################################################################################
# We will separate the contribution of the energy on the different components:
# bond strain, angular strain, dihedral strain, and Van der Waals forces.


def calculate_bond_potential(dist_mat: Matrix) -> Vector:
    """Calculate bond potential due using the harmonic oscillator formula."""

    bond_energies = []

    for index, bond in enumerate(BOND_LIST):
        i, j = bond
        distance = dist_mat[i][j]

        bond_energy = (
            F_CONST[BOND_TYPES[index]]
            * (distance - EQ_DIST[BOND_TYPES[index]]) ** 2
        )
        bond_energies.append(bond_energy)

    return np.array(bond_energies)


def calculate_angle_potential(angle_values: Vector) -> Vector:
    """Calculate angle potential due using the harmonic oscillator formula."""

    angle_energies = []

    for angle, angle_value in enumerate(angle_values):
        angle_energy = (
            F_CONST[ANGLE_TYPES[angle]]
            * (
                angle_value * np.pi / 180
                - EQ_DIST[ANGLE_TYPES[angle]] * np.pi / 180
            )
            ** 2
        )
        angle_energies.append(angle_energy)

    return np.array(angle_energies)


def calculate_dihedral_potential(dih_val: Vector) -> Vector:
    """Calculates the dihedral energies using the formula A_phi(1+cos(n·phi)).

    WARNING the index of the force constant for the dihedral is manually
    placed. Any change to the order of the force constants list will need
    manual adjusting of the index
    """

    dihedral_energies = []
    for dihedral in range(len(DIHEDRAL_LIST)):
        dihedral_value = dih_val[dihedral]
        dihedral_energy = F_CONST[5] * (
            1 + np.cos(EQ_DIST[5] * dihedral_value * np.pi / 180)
        )

        # WARNING the index of the force constant for the dihedral is manually placed. Any change to the order of the
        # force constants list will need manual adjusting of the index

        dihedral_energies.append(dihedral_energy)

    return np.array(dihedral_energies)


def calculate_vdw_potential_matrix(dist_mat: Matrix) -> Matrix:
    """Calculate individual pair VdW energies."""

    N_ATOMS = len(dist_mat)
    vdw_potential_matrix = np.zeros([N_ATOMS, N_ATOMS])

    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):

            i_lj_index = list(WDW_ATOM_LABELS).index(ATOM_LABELS[i])
            j_lj_index = list(WDW_ATOM_LABELS).index(ATOM_LABELS[j])
            # sumar los vectores [i,;] y [:,j] y si el maximo es 2 entonces que no se meta
            if max(CONNECTIVITY_MATRIX[i] + CONNECTIVITY_MATRIX[j]) == 2:
                pass
            elif CONNECTIVITY_MATRIX[i][j] == 0:
                vdw_potential_matrix[i][j] = vdw_potential_matrix[j][i] = (
                    LJ_A_ij[i_lj_index][j_lj_index] / (dist_mat[i][j] ** 12)
                ) - (LJ_B_ij[i_lj_index][j_lj_index] / (dist_mat[i][j] ** 6))

    return vdw_potential_matrix


def calculate_vdw_potential(vdw_potential_matrix: Matrix) -> float:
    """Sum total VdW energies to obtain the total potential."""

    N_ATOMS = len(vdw_potential_matrix)
    vdw_potential = 0

    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):
            vdw_potential += vdw_potential_matrix[i][j]

    return vdw_potential


def calculate_total_potential(
    bond_energies: Matrix,
    angle_energies: Matrix,
    dihedral_energies: Matrix,
    vdw_potential: float,
) -> float:
    """Calculates the potential using all the potential contributions."""

    return (
        sum(bond_energies)
        + sum(angle_energies)
        + sum(dihedral_energies)
        + vdw_potential
    )


def calculate_potential_from_coords(atom_coord_2d: Matrix) -> float:
    """Computes the total potential of the system from atom positions"""

    # calculate some variables for the calculate and print functions
    r_vector_matrix = calculate_r_vector_matrix(atom_coord_2d)
    dist_mat = calculate_dist_mat(r_vector_matrix)

    angle_values = calculate_angles(r_vector_matrix, dist_mat)
    dih_val = calculate_dihedrals(r_vector_matrix)

    bond_energies = calculate_bond_potential(dist_mat)
    angle_energies = calculate_angle_potential(angle_values)
    dihedral_energies = calculate_dihedral_potential(dih_val)
    vdw_potential_matrix = calculate_vdw_potential_matrix(dist_mat)
    vdw_potential = calculate_vdw_potential(vdw_potential_matrix)
    total_potential = calculate_total_potential(
        bond_energies, angle_energies, dihedral_energies, vdw_potential
    )

    return total_potential


################################################################################
#                                                                              #
#                            Cartesian derivatives                             #
#                                                                              #
################################################################################


def bond_derivative(
    bond_indexes: list[int, int], r_vector_matrix: Matrix
) -> Vector:
    """Returns the derivative dr_ab/dx"""

    r_ba = r_vector_matrix[bond_indexes[0]][bond_indexes[1]]
    return r_ba / np.linalg.norm(r_ba)


def angle_derivative(
    angle: list[int, int, int], r_vector_matrix: Matrix
) -> list[Vector]:
    """Returns the derivative dtheta_abc/dx"""

    d_angle = np.zeros([3, 3])

    A, B, C = angle

    r_ba = r_vector_matrix[B][A]
    r_bc = r_vector_matrix[B][C]

    p = np.cross(r_ba, r_bc)

    d_angle[0] = np.cross(-r_ba, p) / (
        np.linalg.norm(r_ba) ** 2 * np.linalg.norm(p)
    )

    d_angle[1] = np.cross(r_ba, p) / (
        np.linalg.norm(r_ba) ** 2 * np.linalg.norm(p)
    ) + np.cross(-r_bc, p) / (np.linalg.norm(r_bc) ** 2 * np.linalg.norm(p))

    d_angle[2] = np.cross(r_bc, p) / (
        np.linalg.norm(r_bc) ** 2 * np.linalg.norm(p)
    )
    return d_angle


def dihedral_derivative(
    dihedral: list[int, int, int, int], r_vector_matrix: Matrix
) -> Vector:
    """Returns the derivative dphi_abcd/dx"""

    d_dihed = np.zeros([4, 3])
    A, B, C, D = dihedral

    r_ab = r_vector_matrix[B][A]
    r_ac = r_vector_matrix[C][A]
    r_bc = r_vector_matrix[C][B]
    r_bd = r_vector_matrix[D][B]
    r_cd = r_vector_matrix[D][C]

    t = np.cross(r_ab, r_bc)
    u = np.cross(r_bc, r_cd)

    d_dihed[0] = np.cross(
        (np.cross(t, r_bc) / (np.linalg.norm(t) ** 2 * np.linalg.norm(r_bc))),
        r_bc,
    )

    d_dihed[1] += np.cross(
        r_ac,
        (np.cross(t, r_bc) / (np.linalg.norm(t) ** 2 * np.linalg.norm(r_bc))),
    )
    d_dihed[1] += np.cross(
        (np.cross(-u, r_bc) / (np.linalg.norm(u) ** 2 * np.linalg.norm(r_bc))),
        r_cd,
    )

    d_dihed[2] += np.cross(
        (np.cross(t, r_bc) / (np.linalg.norm(t) ** 2 * np.linalg.norm(r_bc))),
        r_ab,
    )
    d_dihed[2] += np.cross(
        r_bd,
        (np.cross(-u, r_bc) / (np.linalg.norm(u) ** 2 * np.linalg.norm(r_bc))),
    )

    d_dihed[3] = np.cross(
        (np.cross(-u, r_bc) / (np.linalg.norm(u) ** 2 * np.linalg.norm(r_bc))),
        r_bc,
    )

    return d_dihed


################################################################################
#                                                                              #
#                         Calculating gradients                                #
#                                                                              #
################################################################################


def analytical_stretching_gradient(
    r_vector_matrix: Matrix, dist_mat: Matrix
) -> Matrix:
    """Computes the stretching gradient analytically"""

    stretching_gradient = np.zeros([len(dist_mat), 3])

    for index, bond in enumerate(BOND_LIST):
        i, j = bond
        bond_contribution = (
            2
            * F_CONST[BOND_TYPES[index]]
            * (dist_mat[i][j] - EQ_DIST[BOND_TYPES[index]])
            * bond_derivative(bond, r_vector_matrix)
        )

        stretching_gradient[i] += bond_contribution
        stretching_gradient[j] -= bond_contribution

    return stretching_gradient


def analytical_bending_gradient(
    r_vector_matrix: Matrix, angle_values: Vector
) -> Matrix:
    """Computes the bending gradient analytically"""

    angle_gradient = np.zeros([len(r_vector_matrix), 3])

    for index, angle in enumerate(ANGLE_LIST):
        i, j, k = angle

        angle_contribution = (
            2
            * F_CONST[ANGLE_TYPES[index]]
            * (
                angle_values[index] * np.pi / 180
                - EQ_DIST[ANGLE_TYPES[index]] * np.pi / 180
            )
            * angle_derivative(angle, r_vector_matrix)
        )

        angle_gradient[i] += angle_contribution[0]
        angle_gradient[j] += angle_contribution[1]
        angle_gradient[k] += angle_contribution[2]

    return angle_gradient


def analytical_dihedral_gradient(
    dist_mat: Matrix, r_vector_matrix: Matrix, dihedral_values: Vector
) -> Matrix:
    """Computes the dihedral gradient anallytically"""

    dihedral_gradient = np.zeros([len(dist_mat), 3])

    for index, dihedral in enumerate(DIHEDRAL_LIST):
        i, j, k, l = dihedral

        dihedral_contribution = (
            EQ_DIST[5]
            * F_CONST[5]
            * np.sin(EQ_DIST[5] * dihedral_values[index] * np.pi / 180)
            * dihedral_derivative(dihedral, r_vector_matrix)
        )

        dihedral_gradient[i] += dihedral_contribution[0]
        dihedral_gradient[j] += dihedral_contribution[1]
        dihedral_gradient[k] += dihedral_contribution[2]
        dihedral_gradient[l] += dihedral_contribution[3]

    return dihedral_gradient


def analytical_vdw_gradient(atom_coord_2d: Matrix, dist_mat: Matrix) -> Matrix:
    """Computes the WdW gradient"""

    vdw_gradient = np.zeros([N_ATOMS, 3])

    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):
            if (
                max(CONNECTIVITY_MATRIX[i] + CONNECTIVITY_MATRIX[j]) < 2
                and CONNECTIVITY_MATRIX[i][j] == 0
            ):
                i_lj_index = list(WDW_ATOM_LABELS).index(ATOM_LABELS[i])
                j_lj_index = list(WDW_ATOM_LABELS).index(ATOM_LABELS[j])

                vdw_gradient[i][0:3] += (
                    -12
                    * LJ_A_ij[i_lj_index][j_lj_index]
                    / (dist_mat[i][j] ** 14)
                    + 6
                    * LJ_B_ij[i_lj_index][j_lj_index]
                    / (dist_mat[i][j] ** 8)
                ) * (atom_coord_2d[i] - atom_coord_2d[j])
                vdw_gradient[j][0:3] -= (
                    -12
                    * LJ_A_ij[i_lj_index][j_lj_index]
                    / (dist_mat[i][j] ** 14)
                    + 6
                    * LJ_B_ij[i_lj_index][j_lj_index]
                    / (dist_mat[i][j] ** 8)
                ) * (atom_coord_2d[i] - atom_coord_2d[j])

    return vdw_gradient


def calculate_gradient(
    atom_coord_2d: Matrix, r_vector_matrix: Matrix, dist_mat: Matrix
) -> list[Matrix]:
    """Computes all gradient components and returns the total gradient"""

    angle_values = calculate_angles(r_vector_matrix, dist_mat)
    dih_val = calculate_dihedrals(r_vector_matrix)

    stretching_gradient = analytical_stretching_gradient(
        r_vector_matrix, dist_mat
    )

    angle_gradient = analytical_bending_gradient(r_vector_matrix, angle_values)

    dihedral_gradient = analytical_dihedral_gradient(
        dist_mat, r_vector_matrix, dih_val
    )

    vdw_gradient = analytical_vdw_gradient(atom_coord_2d, dist_mat)

    total_gradient = (
        stretching_gradient + angle_gradient + dihedral_gradient + vdw_gradient
    )

    gradients = [
        stretching_gradient,
        angle_gradient,
        dihedral_gradient,
        vdw_gradient,
        total_gradient,
    ]

    return gradients
    # we will pack here the values to make the return more simple and will eventually unpack them:


def calculate_gradient_from_cartesian(r_k: Vector) -> list[Matrix]:
    """Auxliary function for readability in the internal optimization"""

    r_vector_matrix = calculate_r_vector_matrix(r_k)
    dist_mat = calculate_dist_mat(r_vector_matrix)

    return np.array(calculate_gradient(r_k, r_vector_matrix, dist_mat))


################################################################################
#                                                                              #
#                     Calculating Wilson B & co                                #
#                                                                              #
################################################################################


def calculate_b_matrix(
    internal_COORDINATE_TYPEs: list[int],
    internal_coordinate_members: list[int],
    atom_coord_2d: Matrix,
    r_vector_matrix: Matrix,
) -> Matrix:
    """Calculates the wilson B matrix. It is a 3*natom by n_internal matrix"""

    atom_coord_1d = np.copy(atom_coord_2d)
    atom_coord_1d = atom_coord_2d.reshape([3 * len(r_vector_matrix)])

    b_matrix = np.zeros([len(internal_COORDINATE_TYPEs), len(atom_coord_1d)])

    for i, internal in enumerate(internal_COORDINATE_TYPEs):
        if internal in [0, 1]:
            A, B = internal_coordinate_members[i]

            d_bond = bond_derivative([A, B], r_vector_matrix)

            b_matrix[i][3 * A : 3 * (A + 1)] = d_bond
            b_matrix[i][3 * B : 3 * (B + 1)] = -d_bond

        elif internal in [2, 3, 4]:
            A, B, C = internal_coordinate_members[i]

            d_ang = angle_derivative([A, B, C], r_vector_matrix)

            b_matrix[i][3 * A : 3 * A + 3] = d_ang[0]
            b_matrix[i][3 * B : 3 * B + 3] = d_ang[1]
            b_matrix[i][3 * C : 3 * C + 3] = d_ang[2]

        elif internal == 99:
            A, B, C, D = internal_coordinate_members[i]

            d_dih = dihedral_derivative([A, B, C, D], r_vector_matrix)

            b_matrix[i][3 * A : 3 * A + 3] = d_dih[0]
            b_matrix[i][3 * B : 3 * B + 3] = d_dih[1]
            b_matrix[i][3 * C : 3 * C + 3] = d_dih[2]
            b_matrix[i][3 * D : 3 * D + 3] = d_dih[3]

    return b_matrix


def calculate_g_matrix(b_matrix: Matrix) -> Matrix:
    """Returns the G matrix"""

    return b_matrix @ b_matrix.T


def calculate_g_inverse(g_matrix: Matrix) -> [Vector, Matrix]:
    """Calculates the eigenvalues of the G matrix and returns the
    generalized inverse of the B matrix"""

    Lambda, V = np.linalg.eig(g_matrix)

    Lambda = np.real(Lambda)

    diag_lambda = np.zeros([len(Lambda), len(Lambda)])

    for i, eigenvalue in enumerate(Lambda):
        if eigenvalue > 0.00001:
            diag_lambda[i][i] = 1 / eigenvalue

    g_minus = V @ diag_lambda @ V.T

    return Lambda, g_minus


def calculate_internal_matrices(
    atom_coord_2d: Matrix, r_vector_matrix: Matrix
) -> [Matrix, Matrix, Vector, Matrix]:
    """Calculates the matrices related to the internal optimization"""

    b_matrix = calculate_b_matrix(
        INTERNAL_COORDINATE_TYPES,
        INTERNAL_COORDINATE_MEMEBERS,
        atom_coord_2d,
        r_vector_matrix,
    )
    g_matrix = calculate_g_matrix(b_matrix)
    Lambda, g_inverse = calculate_g_inverse(g_matrix)

    g_matrix = np.real(g_matrix)
    g_inverse = np.real(g_inverse)
    Lambda = np.real(Lambda)

    return b_matrix, g_matrix, Lambda, g_inverse


################################################################################
#                                                                              #
#                         Optimization algorithms                              #
#                                                                              #
################################################################################


def BFGS_optimization(
    r_k: Matrix, grad_0: Matrix, thresh: float = 0.001
) -> [Matrix, float]:
    """Perform the optimization in cartesian coordinates.

    It sets a threshold of rms deviation. Returns the energy and the coordinates
    when a stationary point has been found.


    Parameters
    ----------
    r_k : Matrix
        Atomic positions in 2D.
    grad_0 : Matrix
        Gradient in 2D of the initial structure.
    thresh : float, optional
        Threshold of RMS for convergence in the optimization. The default is 0.001.

    Returns
    -------
    [Matrix, float]
        Returns atomic positions at the stationary point in 2D and the energy.

    """
    print_section_title("CARTESIAN BFGS OPTIMIZATION")
    print("RMSD THRESHOLD IS: %.5f" % thresh)

    step = 0
    rms = 2 * thresh

    hessian = np.diag([1 / 300 for i in range(N_ATOMS * 3)])

    while rms >= thresh:
        print(
            "\n####### Geometry optimization cycle number %i #######\n" % step
        )

        if step == 0:
            print("The initial hessian is:")
            for line in hessian:
                print(line)

        else:
            grad_0 = calculate_gradient_from_cartesian(r_k)
            print("The updated hessian is:")
            for line in hessian:
                print(line)

        p_k = (-hessian @ grad_0[-1].reshape(-1)).reshape(
            [-1, 3]
        )  # reshaped to have a "per atom" dimension

        print("\nThe predicted change p_k is:")
        print_atom_information(p_k)

        r_k1, s_k, energy = line_search(
            r_k, p_k, grad_0
        )  # perform line search with wolfe conditions

        print("New structure r_k+1 = r_k + alpha * p_k is:")
        print_atom_information(r_k1)

        print("\n:: TOTAL ENERGY = %.8f kcal · mol^-1" % (energy))

        grad_k_plus_one = calculate_gradient_from_cartesian(r_k1)

        y_k = (grad_k_plus_one[-1] - grad_0[-1]).reshape(-1)
        v_k = hessian @ y_k.reshape(-1)

        print("\nThe cartesian gradient after step %i is:" % step)
        print_gradient_contribution(*grad_k_plus_one[0:4], long=False)

        hessian = update_hessian(hessian, s_k, y_k, v_k)

        rms = sum(i**2 for i in grad_k_plus_one[-1].reshape(-1)) ** 0.5 / (
            (N_ATOMS * 3) ** 0.5
        )

        print(
            "\nThe gradient root mean square deviation is: %.5f (thresh = %.5f)"
            % (rms, thresh)
        )

        r_k = r_k1

        step += 1

    print_section_title("CONVERGENCE HAS BEEN REACHED IN %i CYCLES" % step)

    r_vector_matrix = calculate_r_vector_matrix(r_k1)
    dist_mat = calculate_dist_mat(r_vector_matrix)

    final_energy = calculate_and_print_energy(dist_mat, r_vector_matrix, False)

    print("\nAtomic positions at the stationary point:")

    print_atom_information(r_k, lab=ATOM_LABELS)

    return r_k, final_energy


def line_search(
    r_k: Matrix, p_k: Matrix, grad_0: Matrix, alpha: float = 0.8
) -> [Matrix, Vector, float]:
    """
    Perform line search to scale the displacement in the cartesian optimization.

    When a search direction is obtained, a check is made to determine if the
    scale of the vector is appropriate. It is considered appropriate when the
    Wolfe conditions are met. If they are not met, the vector of the search
    direction is scaled until the condition is met.

    Parameters
    ----------
    r_k : Matrix
        Atomic positions in 2D.
    p_k : Matrix
        Search direction in 2D.
    grad_0 : Matrix
        Gradient of the energy in cartesian coordinates for this step.
    alpha : float, optional
        Starting value of Alpha for the line search. The default is 0.8.

    Returns
    -------
    [Matrix, Vector, float]
        Returns the new atomic positions in 2D, the displacement in 1D and the
        energy at the point where the Wolfe condition is met.

    """
    print("\nA line search has been started:")

    wolfe_is_met = False
    energy_minus_one = calculate_potential_from_coords(r_k)

    while not wolfe_is_met:

        s_k = alpha * p_k
        r_k1 = r_k + s_k

        energy = calculate_potential_from_coords(r_k1)

        print(
            "Energy value alongside direction p_k with alpha = %.5f is: %.8f"
            % (alpha, energy)
        )
        if energy <= energy_minus_one + 0.1 * alpha * np.dot(
            p_k.reshape(-1), grad_0[-1].reshape(-1)
        ):
            print(
                "Wolfe condition %.5f <= %.5f is met.\n"
                % (
                    energy,
                    energy_minus_one
                    + 0.1
                    * alpha
                    * np.dot(
                        p_k.reshape(-1),
                        grad_0[-1].reshape(-1),
                    ),
                )
            )
            wolfe_is_met = True
        else:
            alpha *= 0.8

    return r_k1, s_k.reshape(-1), energy


def internal_BFGS_optimization(
    atom_coord_2d: Matrix,
    r_vector_matrix: Matrix,
    thresh: float = 0.00001,
    rms_thresh: float = 0.02,
    grms_thresh: float = 0.001,
) -> [Matrix, float]:
    """
    Perform the optimization in internal coordinates.

    For this the steps shown in the notes are followed. When the grms value is
    lower than a certain threshold, the optimization is considered converged.
    Returns the energy and the coordinates when a stationary point has been found.

    Parameters
    ----------
    atom_coord_2d : Matrix
        Matrix with the atom positions.
    r_vector_matrix : Matrix
        Vector matrix between atoms.
    thresh : float, optional
        Threshold for the conversion from internal to cartesian coordinates. The default is 0.00001.
    rms_thresh : float, optional
        Thresholf for the rescaling of the gradient in internal coordinates. The default is 0.02.
    grms_thresh : float, optional
        Threshold for the optimization in internal coordinates. The default is 0.001.

    Returns
    -------
    [Matrix, float]
        Returns the stationary point cartesian coordinates in 2D and the total energy at the stationary point.

    """
    print_section_title("BFGS optimization in internal coordinates")
    x_0 = atom_coord_2d
    hessian = initialize_hessian(atom_coord_2d)
    print("The estimated Hessian for the first step is:\n")
    print(hessian)

    grad_iter = 0
    grms = 2 * grms_thresh
    while grms > grms_thresh:
        # 0 calculate/update the hessian
        grad_iter += 1

        print(
            "\n",
            "#" * 25,
            "Geometry optimizaton step %i" % grad_iter,
            "#" * 25,
        )
        # 1 calculate gradient in cartesian, wilson B and g_inverse and grms

        r_vector_matrix = calculate_r_vector_matrix(
            x_0.reshape(-1, 3)
        )  # update r_vector_matrix

        b_matrix, *_, g_inverse = calculate_internal_matrices(
            x_0, r_vector_matrix
        )  # calculate the wilson matrices

        gc_0 = calculate_gradient_from_cartesian(x_0.reshape(-1, 3))[
            -1
        ].reshape(
            -1
        )  # cartesian gradient at the initial geometry

        gq_0 = (
            g_inverse @ b_matrix @ gc_0.reshape(-1)
        )  # internal gradient in the initial geometry

        grms = (sum(p**2 for p in gc_0) / len(gc_0)) ** 0.5  # gradient rms

        print(
            "\nThe RMS value of the gradient is: %.6f (Threshold %.6f)"
            % (grms, grms_thresh)
        )

        # 2 calculate search direction
        p_q = -hessian @ gq_0

        # 3 calculate step rmsd and p rmsd and rescale if required
        rms = (sum(p**2 for p in p_q) / len(p_q)) ** 0.5

        alpha = 1
        if rms > rms_thresh:
            alpha = rms_thresh / rms
            s_q = p_q * alpha
            rms = (sum(p**2 for p in s_q) / len(s_q)) ** 0.5

            print(
                "\nGradient was rescaled to fit the RMS threshold. The new search step direction is (RMS = %.6f):"
                % rms
            )
            print(s_q)

        # 4 define s_q and q_1
        s_q = p_q * alpha

        x_0 = x_0.reshape(-1)
        q_0 = calculate_internal_values(x_0.reshape((-1, 3)))

        q_1 = q_0 + s_q

        # 5 transform from cartesian to internal coordinates
        s_q = q_1 - q_0
        x_1, q_11 = internal_to_cartesian(
            b_matrix, g_inverse, s_q, x_0, q_1, thresh
        )

        print_atom_information(x_1.reshape(-1, 3))

        print("\nNew_set of internals:")
        print(q_11)

        # 6 calculate gradients at the new structure
        r_vector_matrix = calculate_r_vector_matrix(x_1.reshape(-1, 3))

        b_matrix, *_, g_inverse = calculate_internal_matrices(
            x_1.reshape(-1, 3), r_vector_matrix
        )

        print("\nB matrix at the new structure:")
        print(b_matrix)

        print("\nG inverse matrix at the new structure:")
        print(g_inverse)

        gc_1 = calculate_gradient_from_cartesian(x_1.reshape(-1, 3))[
            -1
        ].reshape(-1)

        gq_1 = g_inverse @ b_matrix @ gc_1

        print("\nGradient in terms of internal coordinates:")
        print(gq_1)

        grms = (sum(p**2 for p in gc_1) / len(gc_1)) ** 0.5

        print(
            "\nThe RMS value of the gradient is: %.6f (Threshold %.6f)"
            % (grms, grms_thresh)
        )

        # 7 update the hessian

        s_q = q_11 - q_0

        for i, s in enumerate(s_q):
            if INTERNAL_COORDINATE_TYPES[i] == 99:
                if s > np.pi:
                    s_q[i] -= 2 * np.pi
                elif s < -np.pi:
                    s_q[i] += 2 * np.pi

        y_q = gq_1 - gq_0
        v_q = np.dot(hessian, y_q.T)

        hessian = update_hessian(hessian, s_q, y_q, v_q)

        print("\nThe updated Hessian is:")
        print(hessian)

        # 8 return the energy
        r_vector_matrix = calculate_r_vector_matrix(x_1.reshape(-1, 3))
        dist_mat = calculate_dist_mat(r_vector_matrix)

        calculate_and_print_energy(dist_mat, r_vector_matrix, long=False)

        x_0 = x_1

    print_section_title(
        "GEOMETRY OPTIMIZATION CONVERGED IN %i CYCLES" % grad_iter
    )

    final_energy = calculate_and_print_energy(
        dist_mat, r_vector_matrix, long=False
    )

    print("\nAtomic positions at the stationary point:")

    print_atom_information(x_1.reshape((-1, 3)), lab=ATOM_LABELS)

    return x_0, final_energy


def initialize_hessian(atom_coord_2d: Matrix) -> Matrix:
    """
    Initialize the hessian for internal coordinates optimization with the selected values depending on the type of internal coordinate.

    Parameters
    ----------
    atom_coord_2d : Matrix
        Matrix with the atom positions.

    Returns
    -------
    Matrix
        Returns the hessian matrix.

    """
    hessian = np.zeros(
        [len(INTERNAL_COORDINATE_TYPES), len(INTERNAL_COORDINATE_TYPES)]
    )
    for i, internal in enumerate(INTERNAL_COORDINATE_TYPES):
        if internal in [0, 1]:
            hessian[i][i] = 1 / 600
        elif internal in [2, 3, 4, 5]:
            hessian[i][i] = 1 / 150
        else:
            hessian[i][i] = 1 / 80

    return hessian


def update_hessian(
    hessian_0: Matrix, s_q: Vector, y_q: Vector, v_q: Vector
) -> Matrix:
    """
    Return the updated hessian using the updated Hessian formula.

    Parameters
    ----------
    hessian_0 : Matrix
        Hessian matrix in the previous step.
    s_q : Vector
        S_q vector.
    y_q : Vector
        Y_q vector.
    v_q : Vector
        V_q vector.

    Returns
    -------
    Matrix
        Updated Hessian.

    """
    a = np.outer((s_q @ y_q + y_q @ v_q) * s_q, s_q) / (s_q @ y_q) ** 2
    b = (np.outer(v_q, s_q) + np.outer(s_q, v_q)) / (s_q @ y_q)

    hessian_0 = hessian_0 + a - b

    return hessian_0


def internal_to_cartesian(
    b_matrix: Matrix,
    g_inverse: Matrix,
    s_q: Vector,
    x_0: Vector,
    q_1: Vector,
    thresh: float = 0.00001,
) -> [Vector, Vector]:
    """Transform from internal coordinates to cartesian coordinates via an iterative procedure.

    Parameters
    ----------
    b_matrix : Matrix
        Wilson B matrix.
    g_inverse : Matrix
        Generalized inverse of B.
    s_q : Vector
        Difference between internals and desired internals in vector form.
    x_0 : Vector
        Cartesian coordinates in cartesian form of the previous step.
    q_1 : Vector
        Desired internal coordinates in vector form.
    thresh : float, optional
        Threshold value for the convergence of the conversion from internal to cartesian coordinates.
        The default is 0.00001.

    Returns
    -------
    [Vector, Vector]
        Returns the new cartesian coordinates x_1 and the new set of internals q_11 in vector form.

    """
    max_change = 1
    cartesian_step = 0
    while max_change > thresh:
        #    for i in range(3):
        cartesian_step += 1
        print(
            "\n",
            "=" * 15,
            "Internal to cartesian conversion iteration %i" % cartesian_step,
            "=" * 15,
        )

        if cartesian_step == 1:
            dx = b_matrix.T @ g_inverse @ s_q

            print("\nEstimated change dx is:")
            print(dx)
            x_1 = x_0 + dx

        q_11 = calculate_internal_values(x_1.reshape((-1, 3)))

        print("\nCurrent internals q_11 are:")
        print(q_11)

        s_q = q_1 - q_11

        print("\nDifference between internals and desired internals is:")
        print(s_q)

        for i, s in enumerate(s_q):
            if INTERNAL_COORDINATE_TYPES[i] == 99:
                if s > np.pi:
                    s_q[i] -= 2 * np.pi
                elif s < -np.pi:
                    s_q[i] += 2 * np.pi

        dx = b_matrix.T @ g_inverse @ s_q

        x_11 = x_1 + dx

        max_change = max(x_1 - x_11)

        print("\nmaximum change is", max_change, "(thresh =", thresh, ")")

        print("\ncorresponding cartesians:")
        print(x_11)
        x_1 = np.copy(x_11)

    print("\n", "=" * 15, "Internal to cartesian converged", "=" * 15)

    return x_1, q_11


################################################################################
#                                                                              #
#                             Printing and output                              #
#                                                                              #
################################################################################


def print_section_title(title: str, width=80):
    """
    Print header to separate sections.

    Parameters
    ----------
    title : str
        Section title.
    width : TYPE, optional
        Select witdth of the section title. The default is 80.

    Returns
    -------
    None.

    """
    print("\n\n")
    print("#" * width)
    plus = 0
    if len(title) % 2 == 0:
        plus = 1
    for _ in range(3):
        print("#", " " * (width - 4), "#")
    print(
        "#",
        " " * (int((width - len(title)) / 2) - 2),
        title,
        " " * (int((width - len(title)) / 2) - plus - 3),
        "#",
    )
    for _ in range(3):
        print("#", " " * (width - 4), "#")
    print("#" * width)
    print("\n")


def print_atom_information(atom_coord_2d: Matrix, lab=False):
    """
    Print atom information including the labels and cartesian coordinates.

    Parameters
    ----------
    atom_coord_2d : Matrix
        Atomic coordinates in matrix form.
    lab : list[str], optional
        Labels that will be displayed with each position vector. The default is False.

    Returns
    -------
    None.

    """
    if lab is False:
        lab = EXT_LAB

    for i, values in enumerate(atom_coord_2d):
        x, y, z = values
        print("{:<3} {:9.6f} {:9.6f} {:9.6f}".format(lab[i], x, y, z))


def print_CONNECTIVITY_MATRIX() -> None:
    """
    For printing purposes. Uses the constant CONNECTIVITY_MATRIX and prints it.

    Works with pandas because it has nice printing.

    Returns
    -------
    None.

    """
    df = pd.DataFrame(
        CONNECTIVITY_MATRIX, columns=EXT_LAB, index=EXT_LAB
    ).astype(int)
    print(df)


def print_bond_information(dist_mat: Matrix, potential=False) -> None:
    """
    Print bond information with the current distance matrix and bond list.

    Parameters
    ----------
    dist_mat : Matrix
    potential : np.nandarray, optional
        Variable used to modify the output. If a list containing the potential
        energy of each contribution is fed to the function, it will be printed.
        The default is False.

    Returns
    -------
    None.

    """
    print("\nBond information:\n")
    for index, _ in enumerate(BOND_LIST):
        i, j = BOND_LIST[index]
        if isinstance(potential, np.ndarray):
            print(
                "Bond {:>3}: {:<7}  {:6.4f} A {:10.5f} kcal mol^-1".format(
                    (index + 1),
                    " ".join([EXT_LAB[i], EXT_LAB[j]]),
                    dist_mat[i][j],
                    potential[index],
                )
            )

        else:
            print(
                "Bond {:>3}: {:<7}  {:6.4f} A ".format(
                    (index + 1),
                    " ".join([EXT_LAB[i], EXT_LAB[j]]),
                    dist_mat[i][j],
                )
            )

    if isinstance(potential, np.ndarray):
        print(
            "\nTotal number of bonds %i. Total energy contribution : %.4f kcal · mol^-1"
            % (len(BOND_LIST), sum(potential))
        )
    else:
        print("\nTotal number of bonds %i" % len(BOND_LIST))


def print_angles_information(angle_values: Vector, potential=False) -> None:
    """
    Print angle information in a pretty format

    Parameters
    ----------
    angle_values : Vector
        Vector that contains the floating point values of the different angles.
    potential : np.nandarray, optional
        Variable used to modify the output. If a list containing the potential
        energy of each contribution is fed to the function, it will be printed.
        The default is False.

    Returns
    -------
    None.

    """
    print("\nAngle information:\n")
    for angle, _ in enumerate(ANGLE_LIST):
        i, j, k = ANGLE_LIST[angle]
        if isinstance(potential, np.ndarray):
            print(
                "Angle {:>3}: {:<12}  {:8.2f} deg {:10.5f} kcal mol^-1".format(
                    (angle + 1),
                    " ".join([EXT_LAB[i], EXT_LAB[j], EXT_LAB[k]]),
                    angle_values[angle],
                    potential[angle],
                )
            )
        else:
            print(
                "Angle {:>3}: {:<12}  {:8.2f} deg".format(
                    (angle + 1),
                    " ".join([EXT_LAB[i], EXT_LAB[j], EXT_LAB[k]]),
                    angle_values[angle],
                )
            )
    if isinstance(potential, np.ndarray):
        print(
            "\nTotal number of angles %i. Total energy contribution : %.4f kcal · mol^-1"
            % (len(angle_values), sum(potential))
        )
    else:
        print("\nTotal number of angles %i" % len(ANGLE_LIST))


def print_dihedrals_information(dih_val: Vector, potential=False) -> None:
    """
    Print angle information in a pretty format.

    Parameters
    ----------
    dih_val : Vector
        Vector that contains the floating point values of the different dihedrals.
    potential : np.nandarray, optional
        Variable used to modify the output. If a list containing the potential
        energy of each contribution is fed to the function, it will be printed.
        The default is False.

    Returns
    -------
    None.

    """
    print("\nDihedral information:\n")
    for dihedral, _ in enumerate(DIHEDRAL_LIST):
        i, j, k, l = DIHEDRAL_LIST[dihedral]
        if isinstance(potential, np.ndarray):
            print(
                "Dihedral {:>3}: {:^16} {:8.2f} deg {:10.5f} kcal mol^-1".format(
                    (dihedral + 1),
                    " ".join([EXT_LAB[i], EXT_LAB[j], EXT_LAB[k], EXT_LAB[l]]),
                    dih_val[dihedral],
                    potential[dihedral],
                )
            )

        else:
            print(
                "Dihedral {:>3}: {:^16} {:8.2f} deg".format(
                    (dihedral + 1),
                    " ".join([EXT_LAB[i], EXT_LAB[j], EXT_LAB[k], EXT_LAB[l]]),
                    dih_val[dihedral],
                )
            )

    if isinstance(potential, np.ndarray):
        print(
            "\nTotal number of dihedrals %i. Total energy contribution : %.4f kcal · mol^-1"
            % (len(DIHEDRAL_LIST), sum(potential))
        )
    else:
        print("\nTotal number of dihedrals %i" % len(DIHEDRAL_LIST))


def print_vdw_pairs(
    vdw_potential_matrix: Matrix, dist_mat: Matrix, vdw_potential: float
) -> None:
    """
    Print all VdW pairs in a pretty format.

    Parameters
    ----------
    vdw_potential_matrix : Matrix
        Matrix containing the VdW potential between pairs of atoms.
    dist_mat : Matrix
        Distance matrix between atoms.
    vdw_potential : float
        Total VdW potential.

    Returns
    -------
    None.

    """
    counter = 0
    print("\nVdW repulsion information:\n")
    print("             Atoms     Distance             Energy")
    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):
            counter += 1
            print(
                "Pair {:<3}: {:<9}  {:8.4f} A {:16.5f} kcal mol^-1".format(
                    (counter),
                    " ".join([EXT_LAB[i], EXT_LAB[j]]),
                    dist_mat[i][j],
                    vdw_potential_matrix[i][j],
                )
            )
    print(
        "\nTotal number of unique atom pairss: %i. Total energy: %.4f kcal mol^-1."
        % (counter, vdw_potential)
    )


def print_gradient_contribution(
    stretching_gradient: Matrix,
    angle_gradient: Matrix,
    dihedral_gradient: Matrix,
    vdw_gradient: Matrix,
    long=True,
) -> None:
    """
    Print all stretching contributions to the gradient.

    Parameters
    ----------
    stretching_gradient : Matrix
        2D representation of the contribution of bonds to the gradient in cartesian coordinates.
    angle_gradient : Matrix
        2D representation of the contribution of angles to the gradient in cartesian coordinates.
    dihedral_gradient : Matrix
        2D representation of the contribution of dihedrals to the gradient in cartesian coordinates.
    vdw_gradient : Matrix
        2D representation of the contribution of VdW repulsion to the gradient in cartesian coordinates.
    long : bool, optional
        Large verbose ammount. The default is True.

    Returns
    -------
    None.

    """
    if long:
        print("\nAnalitical gradient of stretching energy:")
        for i, label in enumerate(EXT_LAB):
            g_x, g_y, g_z = stretching_gradient[i]
            print("{:<3} {:9.4f} {:9.4f} {:9.4f}".format(label, g_x, g_y, g_z))

        print("\nAnalitical gradient of bending energy:")
        for i, label in enumerate(EXT_LAB):
            g_x, g_y, g_z = angle_gradient[i]
            print("{:<3} {:9.4f} {:9.4f} {:9.4f}".format(label, g_x, g_y, g_z))

        print("\nAnalitical gradient of torsional energy:")
        for i, label in enumerate(EXT_LAB):
            g_x, g_y, g_z = dihedral_gradient[i]
            print("{:<3} {:9.4f} {:9.4f} {:9.4f}".format(label, g_x, g_y, g_z))

        print("\nAnalitical gradient of VdW energy:")
        for i, label in enumerate(EXT_LAB):
            g_x, g_y, g_z = vdw_gradient[i]
            print("{:<3} {:9.4f} {:9.4f} {:9.4f}".format(label, g_x, g_y, g_z))

    total_gradient = (
        stretching_gradient + angle_gradient + dihedral_gradient + vdw_gradient
    )

    print("\nAnalitical gradient of overall energy:")
    for i, label in enumerate(EXT_LAB):
        g_x, g_y, g_z = total_gradient[i]
        print("{:<3} {:9.4f} {:9.4f} {:9.4f}".format(label, g_x, g_y, g_z))


def calculate_and_print_energy(
    dist_mat: Matrix, r_vector_matrix: Matrix, long: bool = True
) -> float:
    """
    Calculate the total potential energy and prints it in a pretty format.

    Parameters
    ----------
    dist_mat : Matrix
        Distance matrix between atoms.
    r_vector_matrix : Matrix
        Vector matrix between atoms.
    long : bool, optional
       Large verbose ammount. The default is True.

    Returns
    -------
    total_potential : float
        Total potential energy.

    """
    angle_values = calculate_angles(r_vector_matrix, dist_mat)
    dih_val = calculate_dihedrals(r_vector_matrix)

    # get energy information
    bond_energies = calculate_bond_potential(dist_mat)
    angle_energies = calculate_angle_potential(angle_values)
    dihedral_energies = calculate_dihedral_potential(dih_val)
    vdw_potential_matrix = calculate_vdw_potential_matrix(dist_mat)
    vdw_potential = calculate_vdw_potential(vdw_potential_matrix)
    total_potential = calculate_total_potential(
        bond_energies, angle_energies, dihedral_energies, vdw_potential
    )

    if long:
        print(
            "\nEach internal coordinate and their contribution to the energy are:"
        )
        # print the bond, angle, dihedral and VdW components and contribution to the potential
        print_bond_information(dist_mat, bond_energies)
        print_angles_information(angle_values, potential=angle_energies)
        print_dihedrals_information(dih_val, dihedral_energies)
        print_vdw_pairs(vdw_potential_matrix, dist_mat, vdw_potential)

        print(
            "\nEnergy contribution per type is:\nStr: %.4f kcal · mol^-1\nBen: %.4f kcal · mol^-1\nDih: %.4f kcal · mol^-1\nVdW: %.4f kcal · mol^-1"
            % (
                sum(bond_energies),
                sum(angle_energies),
                sum(dihedral_energies),
                vdw_potential,
            )
        )

    print("\n:: TOTAL ENERGY = %.8f kcal · mol^-1" % (total_potential))

    return total_potential


def calculate_and_print_gradient(
    atom_coord_2d: Matrix,
    r_vector_matrix: Matrix,
    dist_mat: Matrix,
    long=True,
) -> None:
    """
    Calculate the gradient energy and prints it in a pretty format.

    Parameters
    ----------
    atom_coord_2d : Matrix
        Atomic coordinates in matrix form.
    r_vector_matrix : Matrix
        Vector matrix between atoms.
    dist_mat : Matrix
        Distance matrix between atoms.
    long : bool, optional
       Large verbose ammount. The default is True.

    Returns
    -------
    None.

    """
    gradients = calculate_gradient(atom_coord_2d, r_vector_matrix, dist_mat)

    print_gradient_contribution(*gradients[0:4], long)


def print_internal_matrices(
    b_matrix: Matrix,
    g_matrix: Matrix,
    Lambda: Vector,
    g_inverse: Matrix,
    cartesian_grad: Matrix,
) -> None:
    """
    Calculate B, G and G Inverse and prints it in a pretty format.

    Parameters
    ----------
    b_matrix : Matrix
        Wilson B matrix.
    g_matrix : Matrix
        G matrix .
    Lambda : Vector
        List of eigenvalues of the G matrix.
    g_inverse : Matrix
        Generalized inverse of B.
    cartesian_grad : Matrix
        Total gradient in cartesian coordinates.

    Returns
    -------
    None.

    """
    print(
        "\nThe Wilson B matrix at this structure is (%i rows %i columns):\n"
        % (len(b_matrix), len(b_matrix[0]))
    )
    for line in b_matrix:
        print(line)
    print(
        "\nThe G matrix (BB.T) at this structure is (%i rows %i columns):\n"
        % (len(g_matrix), len(g_matrix[0]))
    )
    for line in g_matrix:
        print(line)
    print("\nThe eigenvalues of G are:\n")
    print(Lambda)
    print("\nThe Inverse G matrix at this structure is:\n ")
    for line in g_inverse:
        print(line)
    print(
        "\nThe gradient in internal coordinates is (kcal/mol/Angstrom or kcal/mol/radian):\n"
    )
    print(g_inverse @ b_matrix @ cartesian_grad[-1].reshape(-1))


################################################################################
#                                                                              #
#                         Defining the main functions                          #
#                                                                              #
################################################################################


def regular_format(mol2_file: str) -> bool:
    """
    Read the mol2 file, and identify if it has sections or not.

    Parameters
    ----------
    mol2_file : str
        DESCRIPTION.

    Returns
    -------
    bool
        True if there are separators. False othewise.

    """
    f = read_mol2_file(mol2_file)
    return "@" in f[0]


def startup(mol2_file: str) -> list[int, list[str], Vector, Matrix]:
    """
    Extract the basic information contained in the mol2 file.
    The unconventional mol2 file calls the function that reads the
    contents of a mol2 file without @<> separators.

    Parameters
    ----------
    mol2_file : str
        Input file path.

    Returns
    -------
    list
        Returns a list that contains [number_of_atoms, ATOM_LABELS, atom_coord_2d, CONNECTIVITY_MATRIX].
    """
    if regular_format(mol2_file):
        global N_ATOMS
        content_list = read_mol2_file(mol2_file)
        markers = identify_sections(content_list)
        start_end_atoms = get_atoms_section(content_list, markers)
        atom_coord_2d = get_coordinates(content_list, start_end_atoms)
        N_ATOMS = len(atom_coord_2d)
        ATOM_LABELS = get_labels(content_list, start_end_atoms)
        CONNECTIVITY_MATRIX = get_CONNECTIVITY_MATRIX(content_list, markers)

        return [
            len(CONNECTIVITY_MATRIX),
            ATOM_LABELS,
            atom_coord_2d,
            CONNECTIVITY_MATRIX,
        ]

    return read_unconventional_mol2(mol2_file)


def define_constants(mol2_file: str):
    """
    Define all the constants of the system, such as number of atoms, internal coordinate types, bonds...

    Parameters
    ----------
    mol2_file : str
        DESCRIPTION.

    Returns
    -------
    None, but modifies the global scope setting the value of the constants.

    """
    global N_ATOMS, ATOM_LABELS, CONNECTIVITY_MATRIX, EXT_LAB, BOND_LIST, ANGLE_LIST
    global DIHEDRAL_LIST, UN, BOND_TYPES, ANGLE_TYPES, INTERNAL_COORDINATE_MEMEBERS, INTERNAL_COORDINATE_TYPES

    N_ATOMS, ATOM_LABELS, _, CONNECTIVITY_MATRIX = startup(mol2_file)

    EXT_LAB = extend_labels()

    # print the system information

    BOND_LIST = get_BOND_LIST()
    ANGLE_LIST = build_angles2(BOND_LIST)
    DIHEDRAL_LIST = build_dihedrals2(ANGLE_LIST)
    BOND_TYPES, ANGLE_TYPES = identify_internals()
    UN = [ANGLE_LIST, DIHEDRAL_LIST]

    INTERNAL_COORDINATE_MEMEBERS = BOND_LIST + ANGLE_LIST + DIHEDRAL_LIST
    INTERNAL_COORDINATE_TYPES = np.full(len(INTERNAL_COORDINATE_MEMEBERS), 99)

    for i, internal in enumerate(BOND_TYPES + ANGLE_TYPES):
        INTERNAL_COORDINATE_TYPES[i] = internal


def mol1_report(mol2_file: str):
    """
    Generate the report that includes energy and gradient of the molecule.

    Parameters
    ----------
    filename : str
        Input file path.

    Returns
    -------
    None.

    """
    with open(
        mol2_file.replace(".mol2", "").replace("inputs/", "results/")
        + "_matis.out1",
        "w",
        encoding="utf-8",
    ) as f:
        with redirect_stdout(f):
            print_section_title("Geometry input section")
            atom_coord_2d = startup(mol2_file)[2]

            # print the system information
            print("There are %i atoms in %s file" % (N_ATOMS, mol2_file))
            print_atom_information(atom_coord_2d)
            print("\nThe connectivity matrix of the %s file is" % mol2_file)
            print_CONNECTIVITY_MATRIX()

            print_section_title("Energy calculation")

            # calculate and print energy information
            r_vector_matrix = calculate_r_vector_matrix(atom_coord_2d)
            dist_mat = calculate_dist_mat(r_vector_matrix)
            calculate_and_print_energy(dist_mat, r_vector_matrix)

            print_section_title("Gradient calculation")

            # calculate and print the gradients
            calculate_and_print_gradient(
                atom_coord_2d, r_vector_matrix, dist_mat
            )


def mol2_report(mol2_file: str):
    """
    Generate the report that includes the optimization in cartesian coordinates.

    Parameters
    ----------
    filename : str
        Input file path.

    Returns
    -------
    None.

    """
    with open(
        mol2_file.replace(".mol2", "").replace("inputs/", "results/")
        + "_matis.out2",
        "w",
        encoding="utf-8",
    ) as f:
        with redirect_stdout(f):
            print_section_title("Geometry input section, energy and gradient")
            atom_coord_2d = startup(mol2_file)[2]

            # print the system information
            print(
                "The atoms of the %s system and their positions are:"
                % mol2_file
            )
            print_atom_information(atom_coord_2d)

            # calculate and print energy and gradient information
            r_vector_matrix = calculate_r_vector_matrix(atom_coord_2d)
            dist_mat = calculate_dist_mat(r_vector_matrix)
            calculate_and_print_energy(dist_mat, r_vector_matrix, long=False)
            grad_0 = calculate_gradient_from_cartesian(atom_coord_2d)
            print_gradient_contribution(*grad_0[0:4], False)

            # call the optimization function
            BFGS_optimization(atom_coord_2d, grad_0)


def mol3_report(mol2_file: str):
    """
    Generate the report that includes the optimization in internal coordinates.

    Parameters
    ----------
    filename : str
        Input file path.

    Returns
    -------
    None.

    """
    with open(
        mol2_file.replace(".mol2", "").replace("inputs/", "results/")
        + "_matis.out3",
        "w",
        encoding="utf-8",
    ) as f:

        with redirect_stdout(f):
            print_section_title("Geometry input section, energy and gradient")
            atom_coord_2d = startup(mol2_file)[2]
            r_vector_matrix = calculate_r_vector_matrix(atom_coord_2d)

            print(
                "The atoms of the %s system and their positions are:"
                % mol2_file
            )
            print_atom_information(atom_coord_2d)

            np.set_printoptions(precision=6, linewidth=4000, suppress=True)

            # calculate B, G, G inverse matrix

            b_matrix, g_matrix, Lambda, g_inverse = calculate_internal_matrices(
                atom_coord_2d,
                r_vector_matrix,
            )
            cartesian_grad = calculate_gradient_from_cartesian(atom_coord_2d)

            print_internal_matrices(
                b_matrix, g_matrix, Lambda, g_inverse, cartesian_grad
            )

            internal_BFGS_optimization(atom_coord_2d, r_vector_matrix)


def main(filename: str):
    """
    Call required functions for to each individual report.

    Parameters
    ----------
    filename : str
        Input file path.


    Returns
    -------
    None.

    """
    define_constants(filename)

    a = time.perf_counter()
    mol1_report(filename)
    b = time.perf_counter()
    mol2_report(filename)
    c = time.perf_counter()
    mol3_report(filename)
    d = time.perf_counter()

    print("Time consumption: %.2f, %.2f, %.2f" % (b - a, c - b, d - c))


default_filename = os.getcwd() + "/inputs/methane.mol2"

if __name__ == "__main__":
    # add the possibility to input the filename from the terminal.

    try:
        filename = sys.argv[1]
    except IndexError:
        filename = default_filename

    main(filename)
