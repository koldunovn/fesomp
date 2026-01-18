"""Coordinate transformations between rotated and geographical coordinates.

This module provides functions for converting coordinates and vectors between
rotated (model) coordinates and geographical coordinates using Euler angle
rotations. These transformations are necessary when working with FESOM2 meshes
that use rotated coordinate systems.
"""

from __future__ import annotations

import numpy as np


def _euler_rotation_matrix(
    alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """
    Compute the Euler rotation matrix for coordinate transformations.

    Parameters
    ----------
    alpha : float
        Alpha Euler angle in radians.
    beta : float
        Beta Euler angle in radians.
    gamma : float
        Gamma Euler angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)

    rotate_matrix = np.array([
        [cos_g * cos_a - sin_g * cos_b * sin_a,
         cos_g * sin_a + sin_g * cos_b * cos_a,
         sin_g * sin_b],
        [-sin_g * cos_a - cos_g * cos_b * sin_a,
         -sin_g * sin_a + cos_g * cos_b * cos_a,
         cos_g * sin_b],
        [sin_b * sin_a,
         -sin_b * cos_a,
         cos_b]
    ])

    return rotate_matrix


def scalar_r2g(
    alpha: float,
    beta: float,
    gamma: float,
    rlon: np.ndarray,
    rlat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert coordinates from rotated to geographical coordinate system.

    Parameters
    ----------
    alpha : float
        Alpha Euler angle in degrees.
    beta : float
        Beta Euler angle in degrees.
    gamma : float
        Gamma Euler angle in degrees.
    rlon : np.ndarray
        Longitudes in rotated coordinates (degrees).
    rlat : np.ndarray
        Latitudes in rotated coordinates (degrees).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lon, lat) - Longitudes and latitudes in geographical coordinates (degrees).
    """
    # Convert Euler angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Compute inverse rotation matrix
    rotate_matrix = _euler_rotation_matrix(alpha_rad, beta_rad, gamma_rad)
    rotate_matrix_inv = np.linalg.pinv(rotate_matrix)

    # Convert input coordinates to radians
    rlat_rad = np.deg2rad(rlat)
    rlon_rad = np.deg2rad(rlon)

    # Rotated Cartesian coordinates
    cos_rlat = np.cos(rlat_rad)
    xr = cos_rlat * np.cos(rlon_rad)
    yr = cos_rlat * np.sin(rlon_rad)
    zr = np.sin(rlat_rad)

    # Transform to geographical Cartesian coordinates
    xg = rotate_matrix_inv[0, 0] * xr + rotate_matrix_inv[0, 1] * yr + rotate_matrix_inv[0, 2] * zr
    yg = rotate_matrix_inv[1, 0] * xr + rotate_matrix_inv[1, 1] * yr + rotate_matrix_inv[1, 2] * zr
    zg = rotate_matrix_inv[2, 0] * xr + rotate_matrix_inv[2, 1] * yr + rotate_matrix_inv[2, 2] * zr

    # Convert to geographical coordinates
    lat = np.arcsin(np.clip(zg, -1, 1))
    lon = np.arctan2(yg, xg)

    # Handle points at poles (where x and y are both zero)
    pole_mask = (np.abs(xg) + np.abs(yg)) == 0
    lon = np.where(pole_mask, 0.0, lon)

    return np.rad2deg(lon), np.rad2deg(lat)


def scalar_g2r(
    alpha: float,
    beta: float,
    gamma: float,
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert coordinates from geographical to rotated coordinate system.

    Parameters
    ----------
    alpha : float
        Alpha Euler angle in degrees.
    beta : float
        Beta Euler angle in degrees.
    gamma : float
        Gamma Euler angle in degrees.
    lon : np.ndarray
        Longitudes in geographical coordinates (degrees).
    lat : np.ndarray
        Latitudes in geographical coordinates (degrees).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rlon, rlat) - Longitudes and latitudes in rotated coordinates (degrees).
    """
    # Convert Euler angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Compute rotation matrix (no inverse needed for g2r)
    rotate_matrix = _euler_rotation_matrix(alpha_rad, beta_rad, gamma_rad)

    # Convert input coordinates to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Geographical Cartesian coordinates
    cos_lat = np.cos(lat_rad)
    xg = cos_lat * np.cos(lon_rad)
    yg = cos_lat * np.sin(lon_rad)
    zg = np.sin(lat_rad)

    # Transform to rotated Cartesian coordinates
    xr = rotate_matrix[0, 0] * xg + rotate_matrix[0, 1] * yg + rotate_matrix[0, 2] * zg
    yr = rotate_matrix[1, 0] * xg + rotate_matrix[1, 1] * yg + rotate_matrix[1, 2] * zg
    zr = rotate_matrix[2, 0] * xg + rotate_matrix[2, 1] * yg + rotate_matrix[2, 2] * zg

    # Convert to rotated coordinates
    rlat = np.arcsin(np.clip(zr, -1, 1))
    rlon = np.arctan2(yr, xr)

    # Handle points at poles
    pole_mask = (np.abs(xr) + np.abs(yr)) == 0
    rlon = np.where(pole_mask, 0.0, rlon)

    return np.rad2deg(rlon), np.rad2deg(rlat)


def vec_rotate_r2g(
    alpha: float,
    beta: float,
    gamma: float,
    lon: np.ndarray,
    lat: np.ndarray,
    urot: np.ndarray,
    vrot: np.ndarray,
    flag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate vectors from rotated to geographical coordinates.

    Parameters
    ----------
    alpha : float
        Alpha Euler angle in degrees.
    beta : float
        Beta Euler angle in degrees.
    gamma : float
        Gamma Euler angle in degrees.
    lon : np.ndarray
        Longitudes (in rotated or geographical coordinates, see flag).
    lat : np.ndarray
        Latitudes (in rotated or geographical coordinates, see flag).
    urot : np.ndarray
        U (eastward) component of vector in rotated coordinates.
    vrot : np.ndarray
        V (northward) component of vector in rotated coordinates.
    flag : int
        Coordinate system of input lon/lat:
        - 1: lon/lat are in geographical coordinates
        - 0: lon/lat are in rotated coordinates

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u, v) - Vector components in geographical coordinates.
    """
    # Get coordinates in both systems
    if flag == 1:
        rlon, rlat = scalar_g2r(alpha, beta, gamma, lon, lat)
    else:
        rlon, rlat = lon, lat
        lon, lat = scalar_r2g(alpha, beta, gamma, rlon, rlat)

    # Convert Euler angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Compute inverse rotation matrix
    rotate_matrix = _euler_rotation_matrix(alpha_rad, beta_rad, gamma_rad)
    rotate_matrix_inv = np.linalg.pinv(rotate_matrix)

    # Convert coordinates to radians
    rlat_rad = np.deg2rad(rlat)
    rlon_rad = np.deg2rad(rlon)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Transform vector from local rotated coordinates to Cartesian
    cos_rlat = np.cos(rlat_rad)
    sin_rlat = np.sin(rlat_rad)
    cos_rlon = np.cos(rlon_rad)
    sin_rlon = np.sin(rlon_rad)

    txg = -vrot * sin_rlat * cos_rlon - urot * sin_rlon
    tyg = -vrot * sin_rlat * sin_rlon + urot * cos_rlon
    tzg = vrot * cos_rlat

    # Rotate Cartesian vector components
    txr = rotate_matrix_inv[0, 0] * txg + rotate_matrix_inv[0, 1] * tyg + rotate_matrix_inv[0, 2] * tzg
    tyr = rotate_matrix_inv[1, 0] * txg + rotate_matrix_inv[1, 1] * tyg + rotate_matrix_inv[1, 2] * tzg
    tzr = rotate_matrix_inv[2, 0] * txg + rotate_matrix_inv[2, 1] * tyg + rotate_matrix_inv[2, 2] * tzg

    # Transform back to local geographical coordinates
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    v = -sin_lat * cos_lon * txr - sin_lat * sin_lon * tyr + cos_lat * tzr
    u = -sin_lon * txr + cos_lon * tyr

    return np.asarray(u), np.asarray(v)


def vec_rotate_g2r(
    alpha: float,
    beta: float,
    gamma: float,
    lon: np.ndarray,
    lat: np.ndarray,
    ugeo: np.ndarray,
    vgeo: np.ndarray,
    flag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate vectors from geographical to rotated coordinates.

    Parameters
    ----------
    alpha : float
        Alpha Euler angle in degrees.
    beta : float
        Beta Euler angle in degrees.
    gamma : float
        Gamma Euler angle in degrees.
    lon : np.ndarray
        Longitudes (in rotated or geographical coordinates, see flag).
    lat : np.ndarray
        Latitudes (in rotated or geographical coordinates, see flag).
    ugeo : np.ndarray
        U (eastward) component of vector in geographical coordinates.
    vgeo : np.ndarray
        V (northward) component of vector in geographical coordinates.
    flag : int
        Coordinate system of input lon/lat:
        - 1: lon/lat are in geographical coordinates
        - 0: lon/lat are in rotated coordinates

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u, v) - Vector components in rotated coordinates.
    """
    # Get coordinates in both systems
    if flag == 1:
        rlon, rlat = scalar_g2r(alpha, beta, gamma, lon, lat)
    else:
        rlon, rlat = lon, lat
        lon, lat = scalar_r2g(alpha, beta, gamma, rlon, rlat)

    # Convert Euler angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Compute rotation matrix (no inverse needed for g2r)
    rotate_matrix = _euler_rotation_matrix(alpha_rad, beta_rad, gamma_rad)

    # Convert coordinates to radians
    rlat_rad = np.deg2rad(rlat)
    rlon_rad = np.deg2rad(rlon)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Transform vector from local geographical coordinates to Cartesian
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    txg = -vgeo * sin_lat * cos_lon - ugeo * sin_lon
    tyg = -vgeo * sin_lat * sin_lon + ugeo * cos_lon
    tzg = vgeo * cos_lat

    # Rotate Cartesian vector components
    txr = rotate_matrix[0, 0] * txg + rotate_matrix[0, 1] * tyg + rotate_matrix[0, 2] * tzg
    tyr = rotate_matrix[1, 0] * txg + rotate_matrix[1, 1] * tyg + rotate_matrix[1, 2] * tzg
    tzr = rotate_matrix[2, 0] * txg + rotate_matrix[2, 1] * tyg + rotate_matrix[2, 2] * tzg

    # Transform back to local rotated coordinates
    cos_rlat = np.cos(rlat_rad)
    sin_rlat = np.sin(rlat_rad)
    cos_rlon = np.cos(rlon_rad)
    sin_rlon = np.sin(rlon_rad)

    v = -sin_rlat * cos_rlon * txr - sin_rlat * sin_rlon * tyr + cos_rlat * tzr
    u = -sin_rlon * txr + cos_rlon * tyr

    return np.asarray(u), np.asarray(v)
