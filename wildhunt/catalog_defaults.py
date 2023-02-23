#!/usr/bin/env python

""" This module contains default dictionaries used in the wildhunt Catalog
class.

Specifically it defines presets for online cross-matching of catalogs, including
the default columns to be returned from the cross-matching service.

"""

ls_dr9_default_columns = """
      match.objid,
      match.ls_id,
      match.ra,
      match.dec,
      match.flux_g, match.flux_ivar_g,
      match.flux_r, match.flux_ivar_r,
      match.flux_z, match.flux_ivar_z,
      match.flux_w1, match.flux_ivar_w1,
      match.flux_w2, match.flux_ivar_w2,
      match.flux_w4, match.flux_ivar_w3,
      match.flux_w3, match.flux_ivar_w4,
      match.mag_g,
      match.mag_r,
      match.mag_z,
      match.mag_w1,
      match.mag_w2,
      match.mag_w3,
      match.mag_w4,
      match.maskbits,
      match.snr_g,
      match.snr_r,
      match.snr_z,
      match.snr_w1,
      match.snr_w2,
      match.snr_w3,
      match.snr_w4,
      match.wisemask_w1,
      match.wisemask_w2,
      match.type, 
      match.ebv, 
  """


unwise_dr1_default_columns = """
    match.ra,
    match.dec,
    match.flux_w1,
    match.flux_w2,
    match.flags_unwise_w1,
    match.flags_unwise_w2,
    match.dflux_w1,
    match.dflux_w2,
    match.mag_w1_vg,
    match.mag_w2_vg,
    match.glat,
    match.glon,
"""

catwise2020_default_columns = """
    match.ra,
    match.dec,
    match.w1flux,
    match.w2flux,
    match.w1sigflux,
    match.w2sigflux,
    match.w1mpro,
    match.w2mpro,
    match.w1sigmpro,
    match.w2sigmpro,
    match.w1snr,
    match.w2snr,
    match.glat,
    match.glon,
    match.cc_flags,
    match.ab_flags,
"""

catalog_presets = {
    'DELS': {'service': 'datalab', 'table': 'ls_dr9.tractor',
             'columns': ls_dr9_default_columns},
    'UNWISE': {'service': 'datalab', 'table': 'unwise_dr1.object',
               'columns': unwise_dr1_default_columns},
    'CATWISE': {'service': 'datalab', 'table': 'catwise2020.main',
                'columns': catwise2020_default_columns},
    'UKIDSSDR11LAS': {'service': 'astroquery', 'table': 'ukidssdr11las'},
    'VIKINGDR5': {'service': 'astroquery', 'table': 'vikingdr5'},
}




