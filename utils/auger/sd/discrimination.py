from ...binaries import np
from ...plotting import plt
from ...binaries.binary_tools import pickle_load

class Dataset():

    def __init__(self, /, protons: str, photons: str):
        self.protons = pickle_load(protons)
        self.photons = pickle_load(photons)
        self.scores = set([])


    def __getitem__(self, key):
        match key:
            case "proton" | "protons" | "prot": return self.protons
            case "photon" | "photons" | "phot": return self.photons


    def add_score(self, score: callable, name: str = "score") -> dict:

        for primary in ["protons", "photons"]:
            primary_scores = score(self[primary])
            primary_scores[np.isinf(primary_scores)] = np.nan
            self[primary][name] = primary_scores

        self.scores.add(name)


    def get_templates(self, energy_bins, zenith_bins, score, n_bins=100, show=True, figs=None):

        if score not in self.scores:
            raise ValueError(f"{score} has not been calculated yet!")

        if show:

            if figs is None:
                fig_was_defined = False
                fig1, axes1 = plt.subplots(len(energy_bins)-1, len(zenith_bins)-1, sharex=True, sharey=True)      # for templates
                plt.subplots_adjust(hspace=0.04, wspace=0.02)
                fig2, axes2 = plt.subplots(len(energy_bins)-1, len(zenith_bins)-1, sharex=True, sharey=True)      # for performance
                plt.subplots_adjust(hspace=0.04, wspace=0.02)
            else:
                fig_was_defined = True
                fig1, fig2 = figs
                axes1 = np.reshape(fig1.get_axes(), (len(energy_bins)-1, len(zenith_bins)-1))
                axes2 = np.reshape(fig2.get_axes(), (len(energy_bins)-1, len(zenith_bins)-1))

        vmin = np.min([self["protons"][score].min(), self["photons"][score].min()])
        vmax = np.max([self["protons"][score].max(), self["photons"][score].max()])
        templates = {"proton": {}, "photon": {}}

        for row, (e_low, e_high) in enumerate(zip(energy_bins[:-1], energy_bins[1:])):
            energy_key = f"{np.log10(e_low):.1f}_{np.log10(e_high):.1f}"
            templates["proton"][energy_key] = {}
            templates["photon"][energy_key] = {}

            for col, (t_low, t_high) in enumerate(zip(zenith_bins[:-1], zenith_bins[1:])):
                
                select = lambda df: df[(df["energy"].between(e_low, e_high))
                                    & df["zenith"].between(t_low, t_high)]

                df_protons = select(self["protons"])[score].unique()
                df_photons = select(self["photons"])[score].unique()

                if show:
                    if len(zenith_bins) > 2:
                        template_ax, performance_ax = axes1[row, col], axes2[row, col]
                    else:
                        template_ax, performance_ax = axes1[row], axes2[row]

                    ylabel = f"${np.log10(e_low):.1f} \leq \mathrm{{log}}_{{10}}(E\,/\,\mathrm{{eV}}) \leq {np.log10(e_high):.1f}$"
                    xlabel = fr"${t_low * 180/np.pi:.0f}^\circ\,\leq\,\theta\,\leq\,{t_high * 180/np.pi:.0f}^\circ$"
                

                    n_phot, bins, _ = template_ax.hist(df_photons, range=(vmin, vmax), bins=n_bins, histtype='step',
                                                    label=rf"$\gamma$ ({len(df_photons)} evts)", density=True)
                    n_prot, bins, _ = template_ax.hist(df_protons, range=(vmin, vmax), bins=n_bins, histtype='step',
                                                    label=rf"$p$ ({len(df_protons)} evts)", density=True)

                    template_ax.legend(ncol=2, fontsize=3, loc='upper center',
                                    title=f"{ylabel}, {xlabel}", title_fontsize=3.5)
                    
                    ymin, ymax = template_ax.get_ylim()
                    template_ax.set_ylim(ymin, 1.1*ymax)

                    signal_efficiency = np.cumsum(n_phot) / np.sum(n_phot)
                    bkg_contamination = (np.cumsum(n_prot) / np.sum(n_prot)) / (np.cumsum(n_prot + n_phot) / np.sum(n_prot + n_phot))

                    performance_ax.plot(signal_efficiency, bkg_contamination, marker='none', 
                                        label=score)
                    performance_ax.set_yscale("log")
                
                    if not col:
                        performance_ax.set_ylabel("Bkg. cont.", fontsize=6)
                        template_ax.set_yscale("log")
                        template_ax.set_yticks([])
                        performance_ax.set_yticks([])
                    if not row:
                        x, y = 0.97, 0.05
                        ha, va = 'right', 'bottom'
                    else:
                        x, y = 0.03, 0.95
                        ha, va = 'left', 'top'

                    performance_ax.set_xticks([])
                    performance_ax.set_xlim(0, 1)

                    if not fig_was_defined:
                        performance_ax.text(
                            x, y,
                            f"{ylabel}, {xlabel}",
                            ha=ha, va=va,
                            transform=performance_ax.transAxes,
                            fontsize=3.1
                        )

                    if row == len(energy_bins) - 2:
                        template_ax.set_xlabel("Score", fontsize=6)
                        performance_ax.set_xlabel("Sig. eff.", fontsize=6)

                else:
                    n_phot, bins = np.histogram(df_photons, range=(vmin, vmax), bins=n_bins, density=True)
                    n_prot, bins = np.histogram(df_protons, range=(vmin, vmax), bins=n_bins, density=True)

                zenith_key = f"{t_low * 180/np.pi:.0f}_{t_high * 180/np.pi:.0f}"
                templates["proton"][energy_key][zenith_key] = n_prot
                templates["photon"][energy_key][zenith_key] = n_phot

        templates["score_bins"] = bins

        if show:
            return templates, fig1, fig2
        else:
            return templates