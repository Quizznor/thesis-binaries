from ...binaries import np
from ...plotting import plt

def add_score(data: dict, score: callable, name: str = "score") -> dict:

    data['protons'][name] = score(data["protons"])
    data['photons'][name] = score(data["photons"])

    return data

def get_templates(data, energy_bins, zenith_bins, score, n_bins=100, show=True):

    if show:
        fig1, axes1 = plt.subplots(len(energy_bins)-1, len(zenith_bins)-1, sharex=True, sharey=True)      # for templates
        plt.subplots_adjust(hspace=0.04, wspace=0.02)
        fig2, axes2 = plt.subplots(len(energy_bins)-1, len(zenith_bins)-1, sharex=True, sharey=True)      # for performance
        plt.subplots_adjust(hspace=0.04, wspace=0.02)

    vmin = np.min([data["protons"][score].min(), data["photons"][score].min()])
    vmax = np.max([data["protons"][score].max(), data["photons"][score].max()])
    templates = {}

    for row, (e_low, e_high) in enumerate(zip(energy_bins[:-1], energy_bins[1:])):
        energy_key = f"{np.log10(e_low):.1f}_{np.log10(e_high):.1f}"
        templates[energy_key] = {}

        for col, (t_low, t_high) in enumerate(zip(zenith_bins[:-1], zenith_bins[1:])):
            
            select = lambda df: df[(df["energy"].between(e_low, e_high))
                                   & df["zenith"].between(t_low, t_high)]

            df_protons = select(data["protons"])[score].unique()
            df_photons = select(data["photons"])[score].unique()

            if show:
                if len(zenith_bins) > 2:
                    template_ax, performance_ax = axes1[row, col], axes2[row, col]
                else:
                    template_ax, performance_ax = axes1[row], axes2[row]

                ylabel = f"${np.log10(e_low):.1f} \leq \mathrm{{log}}_{{10}}(E\,/\,\mathrm{{eV}}) \leq {np.log10(e_high):.1f}$"
                xlabel = fr"${t_low * 180/np.pi:.0f}^\circ\,\leq\,\theta\,\leq\,{t_high * 180/np.pi:.0f}^\circ$"
            
                if not col:
                    template_ax.set_yscale("log")
                    template_ax.set_yticks([])
                    performance_ax.set_yticks([])

                n_phot, bins, _ = template_ax.hist(df_photons, range=(vmin, vmax), bins=n_bins, histtype='step',
                                                label=rf"$\gamma$ ({len(df_photons)} evts)", density=True)
                n_prot, bins, _ = template_ax.hist(df_protons, range=(vmin, vmax), bins=n_bins, histtype='step',
                                                label=rf"$p$ ({len(df_protons)} evts)", density=True)

                template_ax.legend(ncol=2, fontsize=6, title=f"{ylabel}, {xlabel}", title_fontsize=5)

                signal_efficiency = np.cumsum(n_phot) / np.sum(n_phot)
                bkg_contamination = (np.cumsum(n_prot) / np.sum(n_prot)) / (np.cumsum(n_prot + n_phot) / np.sum(n_prot + n_phot))

                performance_ax.scatter(signal_efficiency, bkg_contamination)
                performance_ax.set_yscale("log")
            
            else:
                n_phot, bins = np.histogram(df_photons, range=(vmin, vmax), bins=n_bins, density=True)
                n_prot, bins = np.histogram(df_protons, range=(vmin, vmax), bins=n_bins, density=True)

            zenith_key = f"{t_low * 180/np.pi:.0f}_{t_high * 180/np.pi:.0f}"
            templates[energy_key][zenith_key] = {
                "proton": n_prot,
                "photon": n_phot
            }

    templates["bins"] = bins
    return templates
