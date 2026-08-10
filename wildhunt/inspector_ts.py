
import ast
import sys
import gc
import os
import pandas as pd
import math
import numpy as np
from functools import partial

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from wildhunt import utils
from wildhunt import pypmsgs
from wildhunt import image as it

# Suppress warnings!!! THIS IS EXTREMELY DANGEROUS!!!
# I do it anyway :D
import warnings
warnings.filterwarnings("ignore")

# Instantiate message interface
msgs = pypmsgs.Messages()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


default_apertures = {'desdr1': 2.0,
                     'vhsdr6': 2.0,
                     'vikingdr5': 2.0,
                     'unwise-neo3': 6,
                     'unwise-neo2': 6,
                     'unwise-neo4': 6,
                     'unwise-neo5': 6,
                     'unwise-neo6': 6,
                     'PS1': 2.0,
                     'skymapper': 2.0,
                     'vlass': 2.0,
                     'Euclid': 0.9,
                     'HSC': 1.0,
                     'JWST': 1.0  # Placeholder for now
                     }

default_visual_classes = ['point', 'ext', 'badpix', 'blend']


cmaps = matplotlib.pyplot.colormaps()
print(cmaps)

class CutoutViewCanvas(FigureCanvas):

    """A FigureCanvas for showing cutouts from astronomical images.

    This class provides the plotting routines for plotting cutouts for a
    variety of surveys
    """

    def __init__(self, in_dict, parent=None):

        """__init__ method for the CutoutViewCanvas class

        Parameters
        ----------
        parent : obj, optional
            Parent class of SpecFitCanvas
        in_dict : dictionary
            A dictionary containing the input data for plotting
        """

        n_col_bands = in_dict['n_col']
        rgb_bands = in_dict.get('rgb_bands')

        self.n_col = n_col_bands + 1 if rgb_bands else n_col_bands

        n_images = len(in_dict['surveys'])

        self.n_row = int(math.ceil(n_images / n_col_bands))

        self.fig = plt.figure(figsize=(5 * self.n_col, 5 * self.n_row),
                              dpi=80)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.03,
                                 hspace=0.15, wspace=0.25)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.plot(in_dict)



    def plot(self, in_dict):
        """

        :param in_dict:
        :return:
        """

        # Clear axes
        plt.clf()
        gc.collect()

        surveys = in_dict['surveys']
        bands = in_dict['bands']
        fovs = in_dict['fovs']
        apertures = in_dict['apertures']
        square_sizes = in_dict['square_sizes']
        ra = in_dict['ra']
        dec = in_dict['dec']
        image_path = in_dict['image_path']
        verbosity = in_dict['verbosity']
        color_map_name = in_dict['color_map_name']
        color_scale = in_dict.get('color_scale', 'sigma_clip')
        n_sigma = in_dict['n_sigma']


        mag_list = in_dict['mag_list']
        magerr_list = in_dict['magerr_list']
        sn_list = in_dict['sn_list']

        forced_mag_list = in_dict['f_mag_list']
        forced_magerr_list = in_dict['f_magerr_list']
        forced_sn_list = in_dict['f_sn_list']

        flux_list = in_dict['flux_list']
        fluxerr_list = in_dict['fluxerr_list']


        self.fig = it._make_mult_png_axes(self.fig,
                                           self.n_row,
                                           self.n_col,
                                           ra,
                                           dec,
                                           surveys,
                                           bands,
                                           fovs,
                                           apertures,
                                           square_sizes,
                                           image_path,
                                           mag_list,
                                           magerr_list,
                                           sn_list,
                                           forced_mag_list,
                                           forced_magerr_list,
                                           forced_sn_list,
                                           n_sigma=n_sigma,
                                           scalebar=None,
                                           color_map_name=color_map_name,
                                           color_scale=color_scale,
                                           # show_axes=False
                                          )

        rgb_bands = in_dict.get('rgb_bands')
        if rgb_bands:
            rgb_survey = in_dict.get('rgb_survey') or surveys[0]
            rgb_subplot = int('{}{}{}'.format(self.n_row, self.n_col,
                                              len(bands) + 1))
            r_band, g_band, b_band = rgb_bands
            it._make_rgb_axes(self.fig, rgb_subplot, ra, dec, rgb_survey,
                              r_band, g_band, b_band, fovs[0], image_path,
                              n_sigma=n_sigma)

        axes = self.fig.get_axes()
        for i, ax in enumerate(axes[:len(bands)]):
            base_title = f"{surveys[i]} {bands[i]}"
            if flux_list is not None and fluxerr_list is not None:
                flux = flux_list[i]
                fluxerr = fluxerr_list[i]
                if pd.notna(flux) and pd.notna(fluxerr):
                    title = f"{base_title}\n{flux:.2e} ± {fluxerr:.2e}"
                else:
                    title = base_title
            else:
                title = base_title
            ax.set_title(title, fontsize=10)

        for ax in self.fig.get_axes():
            ax.plot(ra, dec, 'o', markerfacecolor='none', markeredgecolor='r', markersize=12, markeredgewidth=1, transform=ax.get_transform('icrs'))

        self.draw()







class ImageViewGUI(QMainWindow):
    def __init__(self, catalog, image_path, ra_column_name,
                 dec_column_name,
                 surveys, bands, psf_size=None, apertures=None,
                 mag_column_names=None, magerr_column_names=None,
                 sn_column_names=None, forced_mag_column_names=None,
                 forced_magerr_column_names=None, forced_sn_column_names=None,
                 auto_download=False, auto_forced_phot=False,
                 minimum_fov=60,
                 visual_classes=None, add_info_list=None, verbosity=0, euclid=False,
                 saved_csv=None, rgb_bands=None, rgb_survey=None):

        QtWidgets.QMainWindow.__init__(self)

        self.setWindowTitle("Visual inspection GUI - Inspector")

        self.df = catalog.copy()

        # Check if visual identification (vis_id) column exists, otherwise
        # create a new one.
        try:
            self.df['vis_id'] = self.df.vis_id.values
        except:
            self.df['vis_id'] = np.nan

        # Check if vis_comment column exists, otherwise create one.
        try:
            self.df['vis_comment'] = self.df.vis_comment.values
        except:
            self.df['vis_comment'] = np.nan

        self.output_filename = saved_csv if saved_csv else f'{surveys[0]}_checked_candidates.csv'
        if saved_csv is not None and os.path.exists(saved_csv):
            old_df = pd.read_csv(saved_csv, dtype={"oid": str})
            if 'oid' in self.df.columns and 'oid' in old_df.columns:
                self.df = self.df.set_index('oid')
                old_df = old_df.set_index('oid')
                self.df.update(old_df[['vis_id', 'vis_comment']])
                self.df = self.df.reset_index()
            else:
                print("Warning: 'oid' column not found. Merging by index instead.")
                self.df.update(old_df)

            remaining = self.df[self.df['vis_id'].isna()]
            if not remaining.empty:
                self.candidate_number = self.df.index.get_loc(remaining.index[0])
            else:
                self.candidate_number = len(self.df) - 1
                print("All objects have been inspected.")
        else:
            self.candidate_number = 0

        # ----------------------------------------------------------------------
        # Set up  class variables
        self.image_path = image_path
        self.ra_column_name = ra_column_name
        self.dec_column_name = dec_column_name
        self.surveys = surveys
        self.bands = bands
        self.rgb_bands = rgb_bands
        self.rgb_survey = rgb_survey or (surveys[0] if surveys else None)
        self.psf_size = psf_size
        self.apertures = apertures
        self.mag_column_names = mag_column_names
        self.magerr_column_names = magerr_column_names
        self.sn_column_names = sn_column_names
        self.forced_mag_column_names = forced_mag_column_names
        self.forced_magerr_column_names = forced_magerr_column_names
        self.forced_sn_column_names = forced_sn_column_names
        self.auto_download = auto_download
        self.auto_forced_photometry = auto_forced_phot
        self.add_info_list = add_info_list
        self.euclid = euclid
        

        if visual_classes is not None:
            if isinstance(visual_classes, dict):
                self.vis_classes = visual_classes

            elif isinstance(visual_classes, list):
                self.vis_classes = {"Classes": visual_classes}

            elif isinstance(visual_classes, str) and visual_classes == 'auto':
                self.vis_classes = default_visual_classes.copy()

            else:
                if verbosity > 0:
                    print('Argument "vis_classes" not understood. Proceeding with default values.')
                self.vis_classes = default_visual_classes.copy()
        else:
            self.vis_classes = default_visual_classes.copy()


        # Check keyword arguments
        if self.apertures is None:
            self.apertures = []
            for idx, survey in enumerate(surveys):
                self.apertures.append(default_apertures[survey])

        # Set up internal interactive defaults
        self.n_col = len(bands)
        self.verbosity = verbosity
        self.color_map_name = 'gray'
        self.color_scale = 'zscale'
        self.n_sigma = 3

        # Setup non-input class variables
        self.mag_list = None
        self.magerr_list = None
        self.sn_list = None
        self.f_mag_list = None
        self.f_magerr_list = None
        self.f_sn_list = None

        self.fovs = [minimum_fov] * len(self.surveys)
        self.square_sizes = [20] * len(self.surveys)

        # Length of catalog file
        self.len_df = self.df.shape[0]
        #self.candidate_number = 0
        self.ind_array = np.arange(self.len_df)

        # Add the menu
        self.create_menu()
        # Add the main widget
        self.main_widget = QtWidgets.QWidget(self)
        # Layout of the main GUI
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        # TODO make startup canvas
        self.update_plot()


        # # Create the mpl Figure and FigCanvas objects.
        self.canvas = CutoutViewCanvas(self.cutout_plot_dict)
        # self.canvas.plot(self.cutout_plot_dict)
        # self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        self.plot_box = QHBoxLayout()
        self.info_layout = QVBoxLayout()
        self.plot_box.addWidget(self.canvas)
        self.plot_box.addLayout(self.info_layout)

        self.create_info_box()

        self.layout.addLayout(self.plot_box)

        # Create the GUI components
        self.create_main_frame()
        self.create_status_bar()

        self.setCentralWidget(self.main_widget)

        # TODO Implement a Finding Chart window
        # TODO Implement a spectral view window



        self.status_dialog = MyDialog(self)



        self.show()

    def update(self):

        # Update input variables
        self.n_sigma = int(self.nsigma_input.text())
        self.fovs = [float(self.clipsize_input.text())] * \
                            len(self.surveys)


        new_color_map = self.cmap_le.text()

        if new_color_map.endswith("_r"):
            new_color_map_name = new_color_map[:-2]
        else:
            new_color_map_name = new_color_map

        print(new_color_map_name)

        if new_color_map_name in cmaps:
            self.color_map_name = new_color_map
        else:
            msgs.warn('Color map name {} is not recognized by '
                      'matplotlib'.format(new_color_map_name))

        new_color_scale = self.colorscale_le.text().strip()

        if new_color_scale in ('sigma_clip', 'zscale'):
            self.color_scale = new_color_scale
        else:
            msgs.warn('Color scale "{}" not recognized. Use "sigma_clip" '
                      'or "zscale".'.format(new_color_scale))

        self.update_plot()

        self.update_info_box()

        self.canvas.plot(self.cutout_plot_dict)



    def update_plot(self):

        idx = self.df.index[self.candidate_number]

        # Update positions
        self.ra = self.df.loc[idx, self.ra_column_name]
        self.dec = self.df.loc[idx, self.dec_column_name]

        self.flux_list = None
        self.fluxerr_list = None
        flux_cols = [f'flux_{band}' for band in self.bands]
        fluxerr_cols = [f'fluxerr_{band}' for band in self.bands]
        if all(col in self.df.columns for col in flux_cols):
            self.flux_list = self.df.loc[idx, flux_cols].values

        if all(col in self.df.columns for col in fluxerr_cols):
            self.fluxerr_list = self.df.loc[idx, fluxerr_cols].values


        # Update catalog magnitudes
        if self.mag_column_names is not None:
            self.mag_list = self.df.loc[idx, self.mag_column_names].values
        if self.magerr_column_names is not None:
            self.magerr_list = self.df.loc[idx, self.magerr_column_names].values
        if self.sn_column_names is not None:
            self.sn_list = self.df.loc[idx, self.sn_column_names].values



        # Update forced photometry
        if self.auto_forced_photometry:
            raise NotImplementedError("Automatic forced photometry of cutouts "
                                      "is not implemented yet.")
        else:
            if self.forced_mag_column_names is not None:
                self.f_mag_list = self.df.loc[idx,
                                         self.forced_mag_column_names].values
            if self.forced_magerr_column_names is not None:
                self.f_magerr_list = self.df.loc[idx,
                                         self.forced_magerr_column_names].values
            if self.forced_sn_column_names is not None:
                self.f_sn_list = self.df.loc[idx,
                                         self.forced_sn_column_names].values

        # Update the cutout plot dictionary (to be passed to CutoutViewCanvas)
        self.cutout_plot_dict = {'ra': self.ra,
                                 'dec': self.dec,
                                 'surveys': self.surveys,
                                 'bands': self.bands,
                                 'fovs': self.fovs,
                                 'apertures': self.apertures,
                                 'square_sizes': self.square_sizes,
                                 'image_path': self.image_path,
                                 'n_col': self.n_col,
                                 'verbosity': self.verbosity,
                                 'color_map_name': self.color_map_name,
                                 'color_scale': self.color_scale,
                                 'n_sigma': self.n_sigma,
                                 'mag_list': self.mag_list,
                                 'magerr_list': self.magerr_list,
                                 'sn_list': self.sn_list,
                                 'f_mag_list': self.f_mag_list,
                                 'f_magerr_list': self.f_magerr_list,
                                 'f_sn_list': self.f_sn_list,
                                 'flux_list': self.flux_list,
                                 'fluxerr_list': self.fluxerr_list,
                                 'rgb_bands': self.rgb_bands,
                                 'rgb_survey': self.rgb_survey}


    def goto_cutout(self):
        self.commit_checkbox_state()
        self.save_data_file()

        new_candidate_number = int(self.goto_le.text())

        if new_candidate_number <= self.len_df-1 and new_candidate_number >= 0:
          self.candidate_number = new_candidate_number

          self.comment_class_le.clear()
          self.update()
          # self.canvas.plot(self.cutout_plot_dict)

        else:
          print('Candidate number invalid!')

    def next_cutout(self):
        self.commit_checkbox_state()
        self.save_data_file()

        if self.candidate_number < self.len_df-1:

          self.candidate_number += 1

          self.comment_class_le.clear()
          self.update()
          # self.canvas.plot(self.cutout_plot_dict)
        else:
          print('Maximum candidate number reached')


    def previous_cutout(self):
        self.commit_checkbox_state()
        self.save_data_file()

        if self.candidate_number > 0:

          self.candidate_number -= 1

          self.comment_class_le.clear()
          self.update()
          # self.canvas.plot(self.cutout_plot_dict)
        else:
          print('Minimum candidate number reached')

    def add_manual_class(self):
        new_class = str(self.manual_class_le.text()).strip()
        if not new_class:
            return

        manual_classes = self.get_manual_classes()
        if new_class not in manual_classes:
            manual_classes.append(new_class)

        self.set_current_vis_dict(self.get_checkbox_vis_dict(), manual_classes)

        if self.verbosity > 1:
            print("Added manual class:", new_class)
            print("Current vis_id:", self.df.loc[self.df.index[self.candidate_number], 'vis_id'])
    
        self.update_info_box()
        self.save_data_file()
        self.manual_class_le.clear()



    def save_comment(self):

        comment = str(self.comment_class_le.text())

        if self.verbosity > 1:
            print('Comment "{}" saved'.format(comment))

        self.df.loc[self.df.index[self.candidate_number], 'vis_comment'] = \
            comment

        self.update_info_box()

    def save_classification(self, classification):

        if self.verbosity > 1:
            print('Class "{}" saved'.format(classification))

        self.df.loc[self.df.index[self.candidate_number], 'vis_id'] = \
            classification

        self.next_cutout()

    def commit_checkbox_state(self):
        self.set_current_vis_dict(self.get_checkbox_vis_dict(),
                                  self.get_manual_classes())


    def save_data_file(self):
        """ Saves the dataframe in a csv table format

        """
        filename = str(self.output_le.text())

        self.df.to_csv(filename, index=False)

        if self.verbosity > 0:
            msgs.info('File with visual classification saved to {}.'.format(
                filename))


    def create_info_box(self):

        idx = self.df.index[self.candidate_number]

        in_dict = self.cutout_plot_dict

        coord_name = utils.coord_to_name(np.array([in_dict['ra']]),
                                      np.array([in_dict['dec']]),
                                      epoch='J')


        self.target_lbl = QLabel('Object {} out of {}'.format(
            self.candidate_number, self.len_df))
        self.coord_name_lbl = QLabel(coord_name[0])

        vis_comment = str(self.df.loc[idx, 'vis_comment'])
        self.visual_comment_label = QLabel('Comment: {'
                                                  '}'.format(vis_comment))

        info_list = [self.target_lbl, self.coord_name_lbl,
                  self.visual_comment_label]

        if self.euclid:
            oid = self.df.loc[idx, 'oid']
            self.oid_lbl = QLabel('oid: {}'.format(oid))
            info_list.append(self.oid_lbl)
            mag_H = self.df.loc[idx, 'mag_H']
            self.mag_H_lbl = QLabel('mag_H: {:.3f}'.format(mag_H))
            info_list.append(self.mag_H_lbl)

        for w in info_list:

            self.info_layout.addWidget(w)

        if self.add_info_list is not None:

            # Adding information from add_info_list to info_box
            self.add_info_Qlabel_list = []
            for add_info in self.add_info_list:

                add_info_type = add_info[0]
                add_info_label = add_info[1]
                add_info_value = None

                if add_info_type == 'column':
                    add_info_value = self.df.loc[idx, add_info[2]]
                elif add_info_type == 'color':

                    # Create magnitude column in df
                    mag_a = self.df[add_info[2]]
                    mag_b = self.df[add_info[3]]
                    self.df[add_info_label] = mag_a - mag_b

                    add_info_value = self.df.loc[idx, add_info_label]
                    add_info_value = '{:.4f}'.format(add_info_value)

                else:
                    raise ValueError("add_info_type {} not recognized.".format(add_info_type))


                add_info_Qlabel = QLabel(str(add_info_label)+': {}'.format(
                add_info_value))
                self.add_info_Qlabel_list.append(add_info_Qlabel)

            for w in self.add_info_Qlabel_list:
                self.info_layout.addWidget(w)

    def update_info_box(self):
        idx = self.df.index[self.candidate_number]

        in_dict = self.cutout_plot_dict

        coord_name = utils.coord_to_name(
            np.array([in_dict['ra']]),
            np.array([in_dict['dec']]),
            epoch='J'
        )

        self.target_lbl.setText('Object {} out of {}'.format(
            self.candidate_number, self.len_df))
        self.coord_name_lbl.setText(coord_name[0])

        vis_comment = str(self.df.loc[idx, 'vis_comment'])
        self.visual_comment_label.setText(
            'Comment: {}'.format(vis_comment)
        )

        # Restore checkbox states from vis_id, matched per category so
        # that identical option labels in different groups (e.g. the
        # 'yes'/'no' options shared by 'Affects photometry' and
        # 'Compactness') don't leak into each other.
        if hasattr(self, 'checkbox_dict'):
            vis_dict = self.get_current_vis_dict()

            for parent, checkboxes in self.checkbox_dict.items():
                selected = [c.strip() for c in
                           vis_dict.get(parent, '').split(",") if c.strip()]
                for checkbox in checkboxes:
                    self.set_button_checked(checkbox,
                                            checkbox.text() in selected)

        if self.euclid:
            oid = self.df.loc[idx, 'oid']
            self.oid_lbl.setText('oid: {}'.format(oid))
            mag_H = self.df.loc[idx, 'mag_H']
            self.mag_H_lbl.setText('mag_H: {:.1f}'.format(mag_H))
            

        if self.add_info_list is not None:

            # Updating add_info_list values
            for jdx, add_info in enumerate(self.add_info_list):
                add_info_type = add_info[0]
                add_info_label = add_info[1]
                add_info_value = None

                if add_info_type == 'column':
                    add_info_value = self.df.loc[idx, add_info[2]]
                elif add_info_type == 'color':
                    add_info_value = self.df.loc[idx, add_info_label]
                    add_info_value = '{:.4f}'.format(add_info_value)

                add_info_Qlabel = self.add_info_Qlabel_list[jdx]

                add_info_Qlabel.setText(
                    str(add_info_label) + ': {}'.format(add_info_value)
                )


    def get_checkbox_vis_dict(self):
        """ Read the currently checked/selected widgets into a
        {category: comma-separated labels} dict.
        """
        vis_dict = {}
        for parent, checkboxes in self.checkbox_dict.items():
            selected = [c.text() for c in checkboxes if c.isChecked()]
            if selected:
                vis_dict[parent] = ",".join(selected)
        return vis_dict

    def set_button_checked(self, widget, state):
        """ Set a checkbox/radio button's checked state.

        QRadioButton refuses to go to "none checked" via a plain
        setChecked(False) call on the currently checked button in an
        auto-exclusive group (Qt keeps at least one radio button checked
        once one has been set). Temporarily disabling autoExclusive
        works around this so classifications actually reset when moving
        to a candidate with no stored selection for that category.
        """
        if isinstance(widget, QRadioButton) and not state:
            was_exclusive = widget.autoExclusive()
            widget.setAutoExclusive(False)
            widget.setChecked(False)
            widget.setAutoExclusive(was_exclusive)
        else:
            widget.setChecked(state)


    def create_main_frame(self):

        self.cmap_lbl = QLabel("Color map:")
        self.cmap_le = QLineEdit(self.color_map_name)
        self.cmap_le.setMaxLength(20)
        self.cmap_le.returnPressed.connect(self.update)

        self.visid_lbl = QLabel("Visual identification:")
        self.visid_le = QLineEdit('new vis_id')
        self.visid_le.setMaxLength(15)

        self.clipsize_lbl = QLabel("Clipsize in arcsec:")
        self.clipsize_input = QLineEdit(str(self.fovs[0]))
        self.clipsize_input.setMaxLength(3)
        self.clipsize_input.returnPressed.connect(self.update)

        self.nsigma_lbl = QLabel("Color scale # of std:")
        self.nsigma_input = QLineEdit(str(self.n_sigma))
        self.nsigma_input.setMaxLength(3)
        self.nsigma_input.returnPressed.connect(self.update)

        self.colorscale_lbl = QLabel("Color scale (sigma_clip/zscale):")
        self.colorscale_le = QLineEdit(self.color_scale)
        self.colorscale_le.setMaxLength(15)
        self.colorscale_le.returnPressed.connect(self.update)

        self.output_lbl = QLabel("Output filename:")
        self.output_le = QLineEdit(self.output_filename)
        
        self.output_le.setMaxLength(40)

        self.goto_le = QLineEdit('1')
        self.goto_le.setMaxLength(4)

        self.save_file_button = QPushButton("Save data file")
        self.save_file_button.clicked.connect(self.save_data_file)
        self.checkbox_dict = {}

        # Create the classification buttons
        class_group_layout = QVBoxLayout()
        group_boxes = {}
        for parent, spec in self.vis_classes.items():
            if isinstance(spec, dict):
                subclasses = spec.get('options', [])
                exclusive = spec.get('exclusive', False)
            else:
                subclasses = spec
                exclusive = False

            group_box = QGroupBox(parent)
            row_layout = QHBoxLayout()
            self.checkbox_dict[parent] = []
            extended_widget = None
            for subclass in subclasses:
                widget = QRadioButton(subclass) if exclusive else QCheckBox(subclass)
                row_layout.addWidget(widget)
                self.checkbox_dict[parent].append(widget)
                if parent == 'Photometry' and subclass == 'extended':
                    extended_widget = widget

            # 'diffused' is an optional sub-selection of Photometry, only
            # relevant (and shown) when 'extended' is selected.
            if extended_widget is not None:
                diffused_checkbox = QCheckBox('diffused')
                diffused_checkbox.setVisible(False)
                row_layout.addWidget(diffused_checkbox)
                self.checkbox_dict[parent].append(diffused_checkbox)

                def _on_extended_toggled(checked, cb=diffused_checkbox):
                    cb.setVisible(checked)
                    if not checked:
                        cb.setChecked(False)

                extended_widget.toggled.connect(_on_extended_toggled)

            row_layout.addStretch()
            group_box.setLayout(row_layout)
            group_boxes[parent] = group_box

        # Show 'Artifact' and 'Affects photometry' side by side as two
        # columns of the same row, if both are present.
        if 'Artifact' in group_boxes and 'Affects photometry' in group_boxes:
            artifact_row = QHBoxLayout()
            artifact_row.addWidget(group_boxes.pop('Artifact'))
            artifact_row.addWidget(group_boxes.pop('Affects photometry'))
            class_group_layout.addLayout(artifact_row)

        for group_box in group_boxes.values():
            class_group_layout.addWidget(group_box)

        self.save_checkbox_button = QPushButton("Save classification")
        self.save_checkbox_button.clicked.connect(self.save_multi_classification)


        self.manual_class_le = QLineEdit('manual classification')
        self.manual_class_le.setMaxLength(22)
        self.manual_class_add_button = QPushButton("Add manual class")
        self.manual_class_add_button.clicked.connect(self.add_manual_class)

        # Add comment box and option to save comment
        self.comment_class_save_button = QPushButton("Save comment")
        self.comment_class_save_button.clicked.connect(
            self.save_comment)
        self.comment_class_le = QLineEdit('visual classification comment')
        self.comment_class_le.setMaxLength(30)
        self.comment_class_le.returnPressed.connect(
            self.comment_class_save_button.click)



        self.previous_button = QPushButton("Previous Cutout")
        self.previous_button.clicked.connect(self.previous_cutout)

        self.next_button = QPushButton("Next Cutout")
        self.next_button.clicked.connect(self.next_cutout)

        self.goto_button = QPushButton("Go to")
        self.goto_button.clicked.connect(self.goto_cutout)

        self.redraw_button = QPushButton("Redraw")
        self.redraw_button.clicked.connect(self.update)

        # self.download_button = QPushButton("Full Download")
        # self.download_button.clicked.connect(self.download_all_image_data)

        # Layout with boxes
        hbox = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()

        for w in [self.redraw_button, self.previous_button, self.next_button,
                  self.goto_button, self.goto_le]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)

        for w in [self.clipsize_lbl, self.clipsize_input,
                  self.cmap_lbl, self.cmap_le, self.nsigma_lbl,
                  self.nsigma_input, self.colorscale_lbl,
                  self.colorscale_le]:
            hbox2.addWidget(w)
            hbox2.setAlignment(w, Qt.AlignVCenter)

        hbox3.addLayout(class_group_layout)
        hbox3.addWidget(self.save_checkbox_button)
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(self.manual_class_le)
        manual_layout.addWidget(self.manual_class_add_button)
        hbox3.addLayout(manual_layout)

        for w in [self.comment_class_le,
                  self.comment_class_save_button,
                  self.output_lbl,
                  self.output_le,
                  self.save_file_button]:
            hbox4.addWidget(w)
            hbox4.setAlignment(w, Qt.AlignVCenter)


        self.layout.addLayout(hbox)
        self.layout.addLayout(hbox2)
        self.layout.addLayout(hbox3)
        self.layout.addLayout(hbox4)




    def create_status_bar(self):
        self.status_text = QLabel("Visual inspection GUI developed by "
                                  "Tatsuyuki Sekine as part of the "
                                  "WildHunt package.")
        self.statusBar().addWidget(self.status_text, 1)


    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        # Add action to save plot
        self.save_plot_action = QAction("&Save plot")
        self.save_plot_action.triggered.connect(self.save_plot)
        # Add shortcut to save plot
        self.save_sc = QShortcut(QKeySequence('Ctrl+S'), self)
        self.save_sc.activated.connect(self.save_plot)

        self.add_actions(self.file_menu, [self.save_plot_action])


        # load_file_action = self.create_action("&Save plot",
        #     shortcut="Ctrl+S", slot=self.save_plot,
        #     tip="Save the plot")
        # quit_action = self.create_action("&Quit", slot=self.close,
        #     shortcut="Ctrl+Q", tip="Close the application")

        # self.add_actions(self.file_menu,
        #     (load_file_action, None, quit_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About",
            shortcut='F1', slot=self.on_about,
            tip='About the demo')

        self.add_actions(self.help_menu, (about_action,))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None,
                        icon=None, tip=None, checkable=False,
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        #if slot is not None:
            #self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action

    def save_plot(self):

        # file_choices = "PNG (*.png)|*.png"

        in_dict = self.cutout_plot_dict

        coord_name = utils.coord_to_name(np.array([in_dict['ra']]),
                                         np.array([in_dict['dec']]),
                                         epoch='J')[0]

        filename = './{}.png'.format(coord_name)

        self.canvas.print_figure(filename, dpi=200)
        self.statusBar().showMessage('Saved to %s' % filename, 2000)

    def on_about(self):
        msg = """ Visual inspection GUI - Inspector
        Author: Tatsuyuki Sekine (tatsuyuki.sekine@studium.uni-hamburg.de)
        Last modified: 3/25/26
        """
        QMessageBox.about(self, "About", msg.strip())

    def set_current_vis_dict(self, checkbox_vis_dict, manual_classes=None):
        """ Store the current classification as a {category: value} dict
        in the 'vis_id' column, e.g.
        "{'Artifact': 'hot pixel', 'Compactness': 'yes'}".

        Uses Python's dict repr (single quotes) rather than JSON, so the
        field has no embedded double quotes for the CSV writer to escape
        by doubling.

        :param checkbox_vis_dict: {category: comma-separated labels} for
         the checkbox/radio-button groups, as returned by
         get_checkbox_vis_dict().
        :param manual_classes: List of freeform manual classes, stored
         under the 'Manual' key.
        """
        row_idx = self.df.index[self.candidate_number]

        vis_dict = dict(checkbox_vis_dict)
        if manual_classes:
            vis_dict['Manual'] = ",".join(manual_classes)

        value = str(vis_dict) if vis_dict else ""

        self.df.loc[row_idx, 'vis_id'] = value

    def get_current_vis_dict(self):
        row_idx = self.df.index[self.candidate_number]
        current = self.df.loc[row_idx, 'vis_id']
        if pd.isna(current) or str(current).strip() == "":
            return {}

        try:
            vis_dict = ast.literal_eval(current)
            return vis_dict if isinstance(vis_dict, dict) else {}
        except (ValueError, SyntaxError):
            return {}

    def get_manual_classes(self):
        vis_dict = self.get_current_vis_dict()

        if vis_dict:
            manual = vis_dict.get('Manual', '')
            return [c.strip() for c in manual.split(",") if c.strip()]

        # Backward compatibility with the old flat comma-separated
        # 'vis_id' format used before per-category classification: treat
        # any label not matching a known checkbox/radio option as manual.
        row_idx = self.df.index[self.candidate_number]
        current = self.df.loc[row_idx, 'vis_id']
        if pd.isna(current) or str(current).strip() == "":
            return []

        checkbox_classes = self.get_all_checkbox_classes()
        return [c.strip() for c in str(current).split(",")
               if c.strip() and c.strip() not in checkbox_classes]

    def get_all_checkbox_classes(self):
        all_classes = set()
        for parent, checkboxes in self.checkbox_dict.items():
            for checkbox in checkboxes:
                all_classes.add(checkbox.text())
        return all_classes

    def save_multi_classification(self):
        self.commit_checkbox_state()

        if self.verbosity > 1:
            print(f'Saved classes: {self.df.loc[self.df.index[self.candidate_number], "vis_id"]}')

        self.next_cutout()


class MyDialog(QDialog):
    def __init__(self, parent = ImageViewGUI):
        super(MyDialog, self).__init__(parent)


        self.resize(680, 600)

        self.close = QPushButton()
        self.close.setObjectName("close")
        self.close.setText("Close")



        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)

        self.buttonBox.addButton(self.close, self.buttonBox.ActionRole)

        self.textBrowser = QTextBrowser(self)
        #self.textBrowser.append("This is a QTextBrowser!")

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.buttonBox)


        self.close.clicked.connect(self.close_click)
        #self.connect(self.close, SIGNAL("clicked()"),self.close_click)


    def close_click(self):
        self.reject()



# class FindingChart(QDialog):
#     def __init__(self, parent = AppForm):
#         super(FindingChart, self).__init__(parent)
#
#
#         # a figure instance to plot on
#         self.figure = plt.figure()
#
#         # this is the Canvas Widget that displays the `figure`
#         # it takes the `figure` instance as a parameter to __init__
#         self.canvas = FigureCanvas(self.figure)
#
#         # this is the Navigation widget
#         # it takes the Canvas widget and a parent
#         self.toolbar = NavigationToolbar(self.canvas, self)
#
#
#         # set the layout
#         layout = QVBoxLayout()
#         #layout.addWidget(self.toolbar)
#         layout.addWidget(self.canvas)
#         self.setLayout(layout)
#
#
#
#
#     def updateme(self,img):
#
#         # create an axis
#         ax = self.figure.add_subplot(111)
#
#         # discards the old graph
#         ax.hold(False)
#
#         # plot data
#         ax.imshow(img)
#
#         # refresh canvas
#         self.canvas.draw()


def run(catalog, image_path, ra_column_name,
                 dec_column_name,
                 surveys, bands, psf_size=None, apertures=None,
                 minimum_fov=10,
                 mag_column_names=None, magerr_column_names=None,
                 sn_column_names=None, forced_mag_column_names=None,
                 forced_magerr_column_names=None, forced_sn_column_names=None,
                 auto_download= False, auto_forced_phot=False,
                 visual_classes=None, add_info_list=None, verbosity=0, euclid=False, saved_csv=None,
                 rgb_bands=None, rgb_survey=None):

    app = QApplication(sys.argv)

    form = ImageViewGUI(catalog, image_path, ra_column_name,
                        dec_column_name, surveys, bands, psf_size=psf_size,
                        apertures=apertures,
                        mag_column_names=mag_column_names,
                        forced_mag_column_names=forced_mag_column_names,
                        magerr_column_names=magerr_column_names,
                        sn_column_names=sn_column_names,
                        forced_magerr_column_names=forced_magerr_column_names,
                        forced_sn_column_names=forced_sn_column_names,
                        auto_download=auto_download,
                        auto_forced_phot=auto_forced_phot,
                        minimum_fov=minimum_fov,
                        add_info_list=add_info_list,
                        visual_classes=visual_classes,
                        verbosity=verbosity,
                        euclid=euclid,
                        saved_csv=saved_csv,
                        rgb_bands=rgb_bands,
                        rgb_survey=rgb_survey)

    form.show()

    app.exec_()




