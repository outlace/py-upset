import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import chain, combinations
from functools import partial
from matplotlib.patches import Rectangle, Circle

class UpSetPlot():
    def __init__(self, n_bases, n_data_points, log=False, filtered=False):
        """
        Generates figures and axes.

        :param n_bases: The number of unique instances

        :param n_data_points: The number of histogram bars (nb of unique sets)
        """
        # map queries to graphic properties
        self.query = []
        qu_col = plt.cm.rainbow(np.linspace(.01, .99, len(self.query)))
        self.query2color = dict(zip([frozenset(q) for q in self.query], qu_col))
        self.query2zorder = dict(zip([frozenset(q) for q in self.query], np.arange(len(self.query)) + 1))

        # set standard colors
        self.greys = plt.cm.Greys([.22, .8])

        # figure details
        self.space = 3 if filtered else 7
        self.log = log
        self.filtered = filtered

        # set figure properties
        self.rows = n_bases
        self.cols = n_data_points
        self.x_values, self.y_values = self._create_coordinates(n_bases, n_data_points)
        self.fig, self.ax_intbars, self.ax_intmatrix, \
        self.ax_setsize, self.ax_tablenames, = self._prepare_figure()

        self.standard_graph_settings = {
            'scatter': {
                'alpha': .3,
                'edgecolor': None
            },
            'hist': {
                'histtype': 'stepfilled',
                'alpha': .3,
                'lw': 0
            }
        }

    def _create_coordinates(self, rows, cols):
        """
        Creates the x, y coordinates shared by the main plots.

        :param rows: number of rows of intersection matrix
        :param cols: number of columns of intersection matrix
        :return: arrays with x and y coordinates
        """
        x_values = (np.arange(cols) + 1)
        y_values = (np.arange(rows) + 1)
        return x_values, y_values

    def _prepare_figure(self):
        """
        Prepares the figure, axes (and their grid)

        :return: references to the newly created figure and axes
        """
        fig = plt.figure(figsize=(25, 12.5))
        topgs = gridspec.GridSpec(1, 1)[0, 0]
        fig_cols = self.cols + 0
        fig_rows = self.rows + self.rows * 4

        gs_top = gridspec.GridSpecFromSubplotSpec(fig_rows, fig_cols, subplot_spec=topgs, wspace=.1, hspace=.2)
        # extend below plots to go higher:
        extend_to_top = 4
        setsize_w, setsize_h = 3, self.rows + extend_to_top
        #tablesize_w, tablesize_h = setsize_w + 2, self.rows
        # move hbar plot more to the left (default=2) 3...for small, 7...for large (is now self.space)
        extend_hbar_to_left = self.space
        tablesize_w, tablesize_h = setsize_w + extend_hbar_to_left, self.rows + extend_to_top
        intmatrix_w, intmatrix_h = tablesize_w + self.cols, self.rows + extend_to_top
        intbars_w, intbars_h = tablesize_w + self.cols, self.rows * 3

        ax_setsize = plt.subplot(gs_top[-setsize_h:-1, 0:setsize_w])
        ax_tablenames = plt.subplot(gs_top[-tablesize_h:-1, setsize_w:tablesize_w])
        ax_intmatrix = plt.subplot(gs_top[-intmatrix_h:-1, tablesize_w:intmatrix_w])
        # move lower part of upper bar plot to top
        extend_bar_to_top = 4
        ax_intbars = plt.subplot(gs_top[:self.rows * 3 + extend_bar_to_top, tablesize_w:intbars_w])

        return fig, ax_intbars, ax_intmatrix, ax_setsize, ax_tablenames

    def _color_for_query(self, query):
        """
        Helper function that returns the standard dark grey for non-queried intersections, and the color assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :return: color as length 4 array.
        """
        query_color = self.query2color.get(query, self.greys[1])
        return query_color

    def _zorder_for_query(self, query):
        """
        Helper function that returns 0 for non-queried intersections, and the zorder assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :return: zorder as int.
        """
        query_zorder = self.query2zorder.get(query, 0)
        return query_zorder

    def main_plot(self, set_names, occurance_list, in_sets, out_sets, set_sizes, unique_bases):
        """
        Creates the main graph comprising bar plot of base set sizes, bar plot of intersection sizes and intersection
        matrix.

        :param set_names: array of names of input data frames, sorted (as above)

        :param in_sets: list of lists. Each list represents the available base sets in a that number of instances.

        :param out_sets: list of lists. Complementary set to in_sets.

        :param set_sizes: array of ints. Contains the intersection sizes.

        :return: dictionary containing figure and axes references.
        """
        # calculate sum of elements for lower plot representing the percentage of missing values
        total_nb_elements = sum(occurance_list)
        # Plots horizontal bar plot for base set sizes (lower left plot)
        ylim = self._base_sets_plot(set_sizes, total_nb_elements)
        self._table_names_plot(set_names, ylim)
        # bar plot showing the set sizes (main plot)
        xlim = self._inters_sizes_plot(occurance_list)
        set_row_map = dict(zip(unique_bases, self.y_values))
        # TODO: adapt _inters_matrix to the available data structures
        self._inters_matrix(in_sets, out_sets, xlim, ylim, set_row_map)
        return {'figure': self.fig,
                'intersection_bars': self.ax_intbars,
                'intersection_matrix': self.ax_intmatrix,
                'base_set_size': self.ax_setsize,
                'names': self.ax_tablenames}

    def _table_names_plot(self, sorted_set_names, ylim):
        ax = self.ax_tablenames
        ax.set_ylim(ylim)
        xlim = ax.get_xlim()
        tr = ax.transData.transform
        for i, name in enumerate(sorted_set_names):
            ax.text(x=1,  # (xlim[1]-xlim[0]/2),
                    y=self.y_values[i],
                    s=name,
                    fontsize=12,
                    clip_on=True,
                    va='center',
                    ha='right',
                    transform=ax.transData,
                    family='monospace')

        # if len(self.x_values) > 1:
        # row_width = self.x_values[1] - self.x_values[0]
        # else:
        #     row_width = self.x_values[0]
        #
        # background = plt.cm.Greys([.09])[0]
        #
        # for r, y in enumerate(self.y_values):
        #     if r % 2 == 0:
        #         ax.add_patch(Rectangle((xlim[0], y - row_width / 2), height=row_width,
        #                                width=xlim[1],
        #                                color=background, zorder=0))
        ax.axis('off')


    def _base_sets_plot(self, sorted_set_sizes, total_nb_elements):
        """
        Plots horizontal bar plot for base set sizes.

        :param sorted_sets: list of data frames, sorted according to user's directives.
        :return: Axes.
        """
        ax = self.ax_setsize
        ax.invert_xaxis()
        height = .6
        bar_bottoms = self.y_values

        sorted_set_sizes_ratio = [elem/total_nb_elements  for elem in sorted_set_sizes]

        bars = ax.barh(bar_bottoms, sorted_set_sizes_ratio, height=height, color=self.greys[1])
        for i, v in enumerate(sorted_set_sizes_ratio):
            ax.text(v+0.1, (i+1), '{:.1f}'.format(v*100), color=self.greys[1], size=11, va='center', ha='right')

        self._strip_axes(ax, keep_spines=['bottom'])#, keep_ticklabels=['bottom'])

        ax.set_ylim((height / 2, (ax.get_ylim()[1] + height / 2)))
        xlim = ax.get_xlim()
        gap = max(xlim) / 500.0 * 20
        ax.set_xlim(xlim[0] + gap, xlim[1] - gap)
        xlim = ax.get_xlim()
        ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

        # bracket_height = ax.transData.inverted().transform([(0, 0), (0, ax.get_ylim()[1])])
        # bracket_height = np.abs(bracket_height[1, 1] - bracket_height[0, 1])
        # for i, (x, y) in enumerate(zip([len(x) for x in sorted_sets], self.y_values)):
        # ax.annotate(sorted_set_names[i], rotation=90, ha='right', va='bottom', fontsize=15,
        #                 xy=(x, y), xycoords='data',
        #                 xytext=(-30, 0), textcoords='offset points',
        #                 arrowprops=dict(arrowstyle="-[, widthB=%s"%(bracket_height,),
        #                                 shrinkA=1,
        #                                 shrinkB=3,
        #                                 connectionstyle='arc,angleA=-180, angleB=180, armB=30',
        #                                 ),
        #                 )

        ax.set_xlabel("Percent missing", fontweight='bold', fontsize=10, labelpad=8)

        return ax.get_ylim()

    def _strip_axes(self, ax, keep_spines=None, keep_ticklabels=None):
        """
        Removes spines and tick labels from ax, except those specified by the user.

        :param ax: Axes on which to operate.
        :param keep_spines: Names of spines to keep.
        :param keep_ticklabels: Names of tick labels to keep.

        Possible names are 'left'|'right'|'top'|'bottom'.
        """
        tick_params_dict = {'which': 'both',
                            'bottom': False,
                            'top': False,
                            'left': False,
                            'right': False,
                            'labelbottom': False,
                            'labeltop': False,
                            'labelleft': False,
                            'labelright': False}
        if keep_ticklabels is None:
            keep_ticklabels = []
        if keep_spines is None:
            keep_spines = []
        lab_keys = [(k, "".join(["label", k])) for k in keep_ticklabels]
        for k in lab_keys:
            tick_params_dict[k[0]] = False
            tick_params_dict[k[1]] = False
        ax.tick_params(**tick_params_dict)
        for sname, spine in ax.spines.items():
            if sname not in keep_spines:
                spine.set_visible(False)

    def _inters_sizes_plot(self, set_sizes):
        """
        Plots bar plot for intersection sizes.
        to the user's directives

        :param set_sizes: array of ints. Sorted, likewise.

        :return: Axes
        """
        ax = self.ax_intbars
        width = .5
        self._strip_axes(ax, keep_spines=['left']) #, keep_ticklabels=['left'])

        bar_bottom_left = self.x_values

        # bars should all have the same color (for now)
        bar_colors = [self.greys[1] for val in set_sizes]

        ax.bar(bar_bottom_left, set_sizes, width=width, color=bar_colors, linewidth=0)
        if self.log:
            ax.set_yscale('log')

        ylim = ax.get_ylim()
        label_vertical_gap = (ylim[1] - ylim[0]) / 60

        for x, y in zip(self.x_values, set_sizes):
            if self.log:
                ax.text(x, 1.3*y, "%.5g" % y,
                        rotation=90, ha='center', va='bottom')
            else:
                ax.text(x, y + label_vertical_gap, "%.5g" % y,
                        rotation=90, ha='center', va='bottom')

        if not self.log:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 15))
        else:
            xlim = ax.get_xlim()
            range_labels = range(5,10) if self.filtered else range(1,10,2)
            for i in range_labels:
                ax.text(xlim[1]*0.99, 10**i * 1.3, r'$10^{%i}$' % i, ha='center', va='bottom', color='grey')

        gap = max(ylim) / 500.0 * 20
        if not self.log:
            ax.set_ylim(ylim[0] - gap, ylim[1] + gap)
        ylim = ax.get_ylim()
        ax.spines['left'].set_bounds(ylim[0], ylim[1])

        if self.log:
            ax.yaxis.grid(True, lw=.35, color='grey', ls=':')
        ax.set_axisbelow(True)
        ax.set_ylabel("Intersection size", labelpad=8, fontweight='bold', fontsize=13)

        return ax.get_xlim()

    def _inters_sizes_plot_diff(self, x_vals, diff_hists, set_sizes, y_title):
        """Plots difference of histograms in bars"""
        # preparing axes
        ax = self.ax_intbars
        width = .5
        interpol='sinc'
        self._strip_axes(ax, keep_spines=['left'], keep_ticklabels=['left'])
        # x values in axes
        bar_x_values = self.x_values
        # function constructing "histogram with color coding"
        def gbar(ax, x, y, hists, max_val, width=0.5, bottom=0, interpol='sinc'):
            # bottom: always 0 for histogram
            for i, (left, top) in enumerate(zip(x, y)):
                right = left + width
                hist = (hists[i][::-1])[:,None]
                im = ax.imshow(hist, interpolation=interpol, cmap=plt.cm.coolwarm,
                               extent=(left, right, bottom, top), alpha=0.8, vmin=-1.*max_val, vmax=max_val)
                path = Path([[left, bottom], [left, top], [right, top], [right, bottom], [left, bottom]])
                patch = PathPatch(path, facecolor='none')
                ax.add_patch(patch)
            return im

        n_x = len(bar_x_values)
        fig = figure()
        xmin, xmax = xlim = 0,n_x+0.2
        ymin, ymax = ylim = np.min(x_vals), np.max(x_vals)
        ax = fig.add_subplot(111, xlim=xlim, ylim=ylim, autoscale_on=False)
        # Hide the right and top spines
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.tick_params(which='both', axis='x', bottom=False, labelbottom=False)
        # ax.set_ylabel('RUWE', labelpad=20, size=14)

        y = np.full(shape=bar_x_values.shape,fill_value=ylim[1])
        # plot bar chart
        im = gbar(ax, bar_x_values, y, diff_hists, width=width, bottom=ylim[1], interpol=interpol, max_val=max(np.abs(np.min(hist)),np.abs(np.max(hist))))
        cbar = plt.colorbar(im)
        cbar.set_label('Relative error', size=14)
        ax.set_aspect('auto')

        # display set sizes 
        ylim = ax.get_ylim()
        label_vertical_gap = (ylim[1] - ylim[0]) / 60
        for x, y in zip(self.x_values, set_sizes):
            ax.text(x, y + label_vertical_gap, "%.5g" % y,
                    rotation=90, ha='center', va='bottom')

        gap = max(ylim) / 500.0 * 20
        if not self.log:
            ax.set_ylim(ylim[0] - gap, ylim[1] + gap)
        ax.spines['left'].set_bounds(ylim[0], ylim[1])

        ax.set_axisbelow(True)
        ax.set_ylabel("Intersection size", labelpad=8, fontweight='bold', fontsize=13)
        return ax.get_xlim()


    def _inters_matrix(self, in_sets, out_sets, xlims, ylims, set_row_map):
        """
        Plots intersection matrix.

        :param in_sets
        :param out_sets

        :param xlims: tuple. x limits for the intersection matrix plot.

        :param ylims: tuple. y limits for the intersection matrix plot.

        :param set_row_map: dict. Maps data frames (base sets) names to a row of the intersection matrix

        :return: Axes
        """
        ax = self.ax_intmatrix
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        if len(self.x_values) > 1:
            row_width = self.x_values[1] - self.x_values[0]
        else:
            row_width = self.x_values[0]

        self._strip_axes(ax)

        background = plt.cm.Greys([.09])[0]

        for r, y in enumerate(self.y_values):
            if r % 2 == 0:
                ax.add_patch(Rectangle((xlims[0], y - row_width / 2), height=row_width,
                                       width=xlims[1],
                                       color=background, zorder=0))

        for col_num, (in_s, out_s) in enumerate(zip(in_sets, out_sets)):
            in_y = [set_row_map[s] for s in in_s]
            out_y = [set_row_map[s] for s in out_s]
            # in_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[1]) for y in in_y]
            # out_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[0]) for y in out_y]
            # for c in chain.from_iterable([in_circles, out_circles]):
            # ax.add_patch(c)
            ax.scatter(np.repeat(self.x_values[col_num], len(in_y)), in_y, color=np.tile(self.greys[1], (len(in_y), 1)),s=200)
            ax.scatter(np.repeat(self.x_values[col_num], len(out_y)), out_y, color=self.greys[0], s=200)
            if len(in_y)>0:
                ax.vlines(self.x_values[col_num], min(in_y), max(in_y), lw=3.5, color=self._color_for_query(frozenset(in_s)))

# Histogram preparation
class PrepareHistogram:
    def __init__(self, histogram, start, stop):
        self.hist = histogram
        self.start = start
        self.stop = stop
        self.bins = np.linspace(start=start, stop=stop, num=histogram.shape[0]+1)
        self.rebinned = False
        self.org_len = histogram.shape[0] # to check when rebinning other histogram

    def get_hist(self):
        """Returns histogram and bins; analogous to np.histogram function"""
        return self.hist, self.bins

    def set_range(self, xmin, xmax):
        """Sets the range of the histogram"""
    
    def _get_min_max_idx(self, xlim=None):
        """Returns the bin index to the minimum and maximum non zero bin"""
        if xlim is None:
            print('xlim is None')
            return np.min(np.nonzero(self.hist)), np.max(np.nonzero(self.hist))
        else:
            # here we are concearned about the bin center
            bin_centers = self.get_bin_midpoints()
            # np.argmax(arr>x) finds index in array where value x gets exceeded, 
            # therefore in the xlim[0] case we subtract a bin-length from the xlim[0] to get the right bin
            bin_distance = np.mean(np.diff(bin_centers))
            min_idx = np.argmax(bin_centers>(xlim[0]-bin_distance))
            # if xlim[1] is larger than the maximum value in bin_centers than the lowest bin gets returned
            # to circumvent this, we check if xlim[1] exceeds the maximum of bin_centers
            if xlim[1]>=np.max(bin_centers):
                max_idx = np.max(np.nonzero(self.hist))
            else:
                max_idx = np.argmax(bin_centers>xlim[1])
            return min_idx, max_idx

    def get_bin_midpoints(self):
        """Returns midpoints array for histogram (e.g. for plotting bar chart)"""
        return self.bins[:-1] + np.diff(self.bins)/2
    
    def rebin(self, nbins):
        """Rebin the histogram to a smaller number of bins"""
        if self.rebinned:
            raise ValueError('Trying to rebin for a second time, might be dangerous and is currently omitted!')
        err_msg = 'The input number of bins has to be of type "unsigned int", ' \
                  'smaller than {} and has to divide {} without any remainder'.format(self.hist.shape[0], self.org_len)
        if isinstance(nbins, int):
            if (nbins>self.hist.shape[0]) or (nbins<0) or (self.org_len%nbins!=0):
                raise ValueError(err_msg)
        else:
            raise ValueError(err_msg)
        # save rebin nbins:
        self.rebinned = True
        # rebin histogram
        self.hist = np.sum(self.hist.reshape(nbins, self.hist.size//nbins), axis=1)
        self.bins = np.linspace(start=self.start, stop=self.stop, num=nbins+1)
        return self.hist, self.bins

    def _rebin_hist(self, other_hist):
        """Rebin other histogram to the size of self.hist"""
        err_msg = 'The input histogram has to have a larger number of bins than self.hist ({})'.format(self.hist.shape[0])
        if other_hist.shape[0]<=self.hist.shape[0]:
            raise ValueError(err_msg)
        nbins = self.hist.shape[0]
        # rebin histogram
        other_hist = np.sum(other_hist.reshape(nbins, other_hist.size//nbins), axis=1)
        return other_hist

    # draw n random samples from histogram
    def _draw_from_hist(self, n):
        bin_midpoints = self.get_bin_midpoints()
        cdf = np.cumsum(self.hist)
        cdf = cdf / cdf[-1]
        values = np.random.rand(n)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = bin_midpoints[value_bins]
        return np.histogram(random_from_cdf, bins=self.bins)[0]

    def draw_n_times(self, n_times, n_samples):
        """Draw from the histogram n_samples, and repeat this process (for smoothing) n_times"""
        from tqdm import tqdm
        add_hist = np.zeros(shape=self.hist.shape)
        for i in tqdm(range(n_times)):
            add_hist += self._draw_from_hist(n_samples)
        return add_hist/n_times

    def get_diff(self, sub_hist, log=False, xlim=None):
        """Return a differnce array for each bin in the histogram
                - log: if the positive and negative differences should be considered with the log scale
                       (meaning: differences<1: 0, else: sign(diff)*log(abs(diff))) 
        """
        # first of all check if self.hist has been rebinned; if so rebin sub_hist
        if self.rebinned and self.org_len==sub_hist.shape[0]:
            sub_hist = self._rebin_hist(sub_hist)
        # first draw a sample from the "super" histogram, then compare it to the "sub_hist", now we fix n_times=10
        drawn_hist = self.draw_n_times(n_times=10, n_samples=np.sum(sub_hist))
        # relative difference between expected (=drawn) histogram and sub_hist
        # first check if drawn_hist has non-zero values everywhere sub_hist has also non-zero values 
        #     (make something like laplace correction)
        drawn_hist[(sub_hist!=0) & (drawn_hist==0)] = 1
        diff_h = (sub_hist-drawn_hist)/(drawn_hist)
        # get nan's where sub_hist&drawn_hist=0, fill them with 0's
        diff_h[~np.isfinite(diff_h)] = 0
        if log:
            # special form of log10: in order to cope with negative differences
            diff_h[np.abs(diff_h)<1]=0.
            diff_h[np.abs(diff_h)>=1] = np.sign(diff_h[np.abs(diff_h)>=1])*np.log10(np.abs(diff_h[np.abs(diff_h)>=1]))
        min_idx, max_idx = self._get_min_max_idx(xlim=xlim)
        return diff_h[min_idx:max_idx], self.get_bin_midpoints()[min_idx:max_idx]


class PrepareData:
    def __init__(self, path, base_names, hist_name=None):
        """Initialize the class
            - path (str): path to the base_set array (containing dictionaries)
            - Base_names: list of tuples with [(variable_name_in_base_set, variable_name_representing_this_set), (.,.), ...]
            - hist_name: name of histogram which should be be displayed in main plot as colour code in histogram
        """
        # sort the missing values in descending order
        self.missing_list = sorted(np.load(path), key=lambda k: k['occurance'], reverse=True)
        self.hist_name = None
        if isinstance(hist_name, str):
            base_set_keys = list(self.base_set[0].keys())
            # remove occurance and nan-columns
            base_set_keys.remove('occurance')
            base_set_keys.remove('nan-columns')
            if self.hist_name in base_set_keys:
                self.hist_name = hist_name
        self.remove_nbs = []
        # additionally to returning base_fancy_names, the remove_nbs variable is set
        self.base_fancy_names = self._get_base_set_fancy_names(base_names)


    def _get_nan_cols_list(self):
        """Returns the missing columns per set"""
        return [dic['nan-columns'] for dic in self.missing_list]

    def _get_occurance_list(self):
        """Returns the set occurances"""
        return [dic['occurance'] for dic in self.missing_list]

    def _get_base_sets(self):
        """seperates the differnt bases
                - nan_list (list): list of lists containing nan column names
        """
        list_of_lists = self._get_nan_cols_list()
        list_of_lists.sort(key=len)
        base_sets = []
        for lst in list_of_lists:
            x = set(lst)
            for base in base_sets:
                # some instances are in two base sets: 
                # if statement ensures that the sets are represented as a whole 
                #    -> one value can therefore exist in multiple base sets (which is what we want)
                if len(set(x)-set(base))==(len(x)-len(base)):
                    x -= set(base)
            if len(list(x))>0:
                base_sets.append(list(x))
        # remove duplicates in bases
        new_intersects = []
        for i in range(len(base_sets)):
            for j in range(i+1, len(base_sets)):
                # if there exists an intersection
                intersect = list(set(base_sets[i]).intersection(base_sets[j]))
                if len(intersect)>0:
                    # save it in new_intersects
                    new_intersects.append(intersect)
                    # then remove it from list_of_list[i&j]
                    for item in intersect:
                        base_sets[i].remove(item)
                        base_sets[j].remove(item)
        # keep only unique elements       
        new_intersects = [list(x) for x in set(tuple(x) for x in new_intersects)]
        new_intersects.sort(key=len, reverse=True)
        # append to main list
        for base in new_intersects:
            base_sets.append(base)
        # return only not empty lists
        return [x for x in base_sets if x != []]

    def _get_bases(self, column_list, base_sets):
        """create list of missing bases (bases turned into numbers for easier comparability)"""
        x = set(column_list)
        base_lst = []
        for i, base in enumerate(base_sets):
            if x != x-set(base):
                base_lst.append(i)
        return base_lst

    def _get_list_of_bases_per_subset(self):
        """Returns list of integers where each integer corresponds to the base set from self._get_base_sets()"""
        return [self._get_bases(dic['nan-columns'], self._get_base_sets()) for dic in self.missing_list]

    def _get_invert_list_of_bases_per_subset(self):
        """invert the base list"""
        invert_base_list = []
        for base_l in self._get_list_of_bases_per_subset():
            invert_base_list.append(list(set(self._get_unique_int_bases())-set(base_l)))
        return invert_base_list

    def _get_unique_int_bases(self):
        """Returns the unique (int) bases"""
        return list(set([item for sublist in self._get_list_of_bases_per_subset() for item in sublist]))

    def _get_base_set_fancy_names(self, base_names):
        """Returns a list of 'clean' base names"""
        mapping_nbs_fancy_names = []
        for (var_in_base, fancy_base_str) in base_names:
            for i, base_elem in enumerate(self._get_base_sets()):
                if var_in_base in base_elem:
                    mapping_nbs_fancy_names.append((i, fancy_base_str))
        mapping_nbs_fancy_names = sorted(mapping_nbs_fancy_names, key=lambda k: k[0])
        bases, names = zip(*mapping_nbs_fancy_names)
        # check out which bases are not supported by fancy_base_str
        unique_bases = self._get_unique_int_bases()
        self.remove_nbs.extend(list(set(self._get_unique_int_bases())-set(bases)))
        return names

    def _get_total_base_occurance(self):
        """count how often a certain base is missing (each time a base is in a base list we
        add up that total_base_occurance for that base...histogram for bases)
        """
        total_base_occurance = {}
        none_missing = 0
        for base in self._get_unique_int_bases():
            total_base_occurance[base] = 0
            for (base_l, occ) in zip(self._get_list_of_bases_per_subset(), self._get_occurance_list()):
                if base in base_l:
                    total_base_occurance[base] += occ
                elif len(base_l) == 0:
                    none_missing = occ
        return total_base_occurance

    def get_args_upset_class(self):
        """Returns the main args for the py-upset class init call, i.e.:
            - n_bases=len(unique_bases)
            - n_data_points=len(base_list)
        """
        # 1) get variables:
        unique_bases, base_list = self._get_unique_int_bases(), self._get_list_of_bases_per_subset()
        # 2) remove the unwanted data column:
        # remove feature 8 from the pool of features
        for remove_nb in self.remove_nbs:
            # remove in unique_bases
            if remove_nb in unique_bases:
                unique_bases.remove(remove_nb)
            # remove in base_list
            for i in range(len(base_list)):
                if remove_nb in base_list[i]:
                    base_list[i].remove(remove_nb)
        # return variables
        n_bases, n_data_points = len(unique_bases), len(base_list)
        return n_bases, n_data_points

    def get_kwargs_upset_plot(self):
        """Returns the main args for the py-upset plot function, i.e.:
                - set_names=base_set_names
                - occurance_list=occu_list
                - in_sets=base_list
                - out_sets=invert_base_list
                - set_sizes=base_occurances
                - unique_bases=unique_bases
        """
        # 1) get variables:
        total_base_occurance, occu_list = self._get_total_base_occurance(), self._get_occurance_list()
        unique_bases, base_set_names = self._get_unique_int_bases(), self.base_fancy_names
        base_list, invert_base_list = self._get_list_of_bases_per_subset(), self._get_invert_list_of_bases_per_subset()
        # 2) remove the unwanted data column:
        # remove feature 8 from the pool of features
        for remove_nb in self.remove_nbs:
            # remove in unique_bases
            if remove_nb in unique_bases:
                unique_bases.remove(remove_nb)
            # remove in base_list
            for i in range(len(base_list)):
                if remove_nb in base_list[i]:
                    base_list[i].remove(remove_nb)
                else:
                    invert_base_list[i].remove(remove_nb)
            total_base_occurance.pop(remove_nb, None)
        base_occurances = [total_base_occurance[k] for k in total_base_occurance.keys()]
        # some lists need sorting
        base_occurances_sorted = [base_occurances[idx] for idx in sorted(range(len(base_occurances)), key=lambda k: base_occurances[k], reverse=True)]
        base_set_names_sorted = [base_set_names[idx] for idx in sorted(range(len(base_occurances)), key=lambda k: base_occurances[k], reverse=True)]
        unique_bases_sorted = [unique_bases[idx] for idx in sorted(range(len(base_occurances)), key=lambda k: base_occurances[k], reverse=True)]
        # return kwargs:
        plot_kwargs = {'set_names': base_set_names_sorted, 'occurance_list': occu_list,
                       'in_sets': base_list, 'out_sets': invert_base_list,
                       'set_sizes': base_occurances_sorted, 'unique_bases': unique_bases_sorted}
        return plot_kwargs



# -------- main --------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear', help='Main histogram has linear instead of logarithmic scale', default=False, action='store_true')
    parser.add_argument('--filter', help='Display only the 99.99% most occuring sub sets.', default=False, action='store_true')
    args = parser.parse_args()
    # now the 'real' main can begin
    import sys
    path_missing_dict = '/home/sebastian/Documents/astonomy-python/data-storage/dr2_storage_finished_list_full.npy'
    base_names = [('astrometric_pseudo_colour', 'Astrometric pseudo-color'),
                  ('radial_velocity', 'Radial velocity'),
                  ('a_g_val', 'Extinction/Redening'),
                  ('radius_val', 'Radius/Luminiosity'),
                  ('teff_val', 'Effective Temperature'),
                  ('phot_rp_mean_mag', 'Flux RP band'),
                  ('parallax', 'Parallax/Proper motion'),
                  ('phot_bp_mean_mag', 'Flux BP band') ]
    prep_data = PrepareData(path_missing_dict, base_names)
    # filter currently not implemented in PrepareData
    #if args.filter:
    #    total_nb_entries = sum(occu_list)
    #    print(total_nb_entries)
    #    sum_occ = 0
    #    break_idx = 0
    #    for i, occ in enumerate(occu_list):
    #        print(occ)
    #        sum_occ += occ
    #        if sum_occ>total_nb_entries*0.9999:
    #            break_idx = i+1
    #            break
    #    print('break_idx: {}'.format(break_idx))
    #    base_list = base_list[:break_idx]
    #    occu_list = occu_list[:break_idx]
    n_bases, n_data_points = prep_data.get_args_upset_class()
    upset = UpSetPlot(n_bases=n_bases, n_data_points=n_data_points, log=(not args.linear), filtered=args.filter)
    plot_kwargs = prep_data.get_kwargs_upset_plot()
    print(plot_kwargs)
    fig_dict = upset.main_plot(**plot_kwargs)

    print('Saving figure...')
    plt.savefig('./savefig.png')

