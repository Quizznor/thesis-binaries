from ...binaries import *
from ...plotting import *
from ...CONSTANTS import GPS_OFFSET

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
from utils.auger.fd import PixelPlot
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from scipy.stats import zscore
from astropy.timeseries import LombScargle
import ruptures as rpt

from functools import cache

class Periodogram(LombScargle):
    
    def __init__(self, x, y):
        self.min_frequency = 1/np.max(x)

        super().__init__(x, y)

    @cache
    def power_spectral_density(self):
        freq, power = self.autopower(
                minimum_frequency = 1/np.max(self.t),
                maximum_frequency = 1,
                samples_per_peak = 10,
                normalization = "psd"
        )

        mask = np.argsort(period := 1/freq)
        return period[mask], power[mask] / 1e3
    

    def show(self, ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(*self.power_spectral_density(), marker="none")
        ax.set_xlabel('Period / $\mathrm{day}$')
        ax.set_ylabel(r"PSD / $\frac{\mathrm{ADC}^2}{\mathrm{mHz}}$")
        ax.set_xscale("log")        

class CameraNSB():

    BASE = Path(f"/cr/tempdata01/filip/nsb_data/")

    def __init__(self, site: str, tel: int):

        IGNORE_SAMPLES_BELOW = 40
        df, n = None, 0

        self.source_dir = self.BASE / f"{site}{tel}"
        for file in self.source_dir.iterdir():
            print(f"loading {file}...", end='\r')
            
            df_part = tools.pickle_load(file)
            df_part = df_part[df_part["Samples"] >= IGNORE_SAMPLES_BELOW]
            df_part = df_part.dropna()

            if df is None:
                df = df_part
            else:
                df = pd.concat([df, df_part], ignore_index=True)

            n += 1

        df["Date"] = pd.to_datetime(df["Date"])
        df.rename(columns={"Date": "datetime"}, inplace=True)
        df["days_since_start"] = xt = df["datetime"].apply(lambda x: x.timestamp())
        df["days_since_start"] = (xt - xt.min()) / (24 * 3600)

        if site == "Coihueco" and tel == 4:
            df.rename(columns={"Baseline": "median_Baseline"}, inplace=True)

        df["night_sky_background"] = (df["median_Variance"] - df["median_Baseline"]) / 1e3

        print(f"\n{n} files ({n/12:.1f} yrs) loaded for {site} #{tel}")

        self.df = df
        self.tel = tel
        self.site = site


    @cache
    def get_pixel(self, pixel, remove_outliers=True,
                  verbose=True) -> pd.DataFrame:

        pixel_df = self.df[self.df["PixelId"] == pixel]

        if remove_outliers:
            original_length = len(pixel_df)
            while len(pixel_df["datetime"]) != sum(zscore(pixel_df["night_sky_background"]) < 5):
                pixel_df = pixel_df[zscore(pixel_df["night_sky_background"]) < 5]

            if verbose: print(f"{original_length - len(pixel_df)} outliers removed for {pixel = }")

        return PixelNSB(pixel, self.site, self.tel, pixel_df)
   

    def camera_drift(self, show=True, ignore_pixels=[]) -> np.ndarray:

        percent_drift_per_year = np.zeros(440)
        for pix in tqdm(range(1, 441)):
            if pix in ignore_pixels:
                percent_drift_per_year[pix-1] = np.nan
                continue

            pixel = self.get_pixel(pix, verbose=False)
            xt = pixel.df["days_since_start"].values
            y = pixel.df["night_sky_background"].values

            popt = np.polyfit(xt, y, deg=1)
            f = np.poly1d(popt)
            drift = popt[0]/f(0) * 36500

            percent_drift_per_year[pix-1] = drift

        if show:

            def get_row(row):
                return percent_drift_per_year[row::22]

            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, 
                                            width_ratios=[0.2, 0.8])

            data_max = np.nanmax(np.abs(percent_drift_per_year))
            vmin, vmax = np.floor(-data_max), np.ceil(data_max)

            ax2 = PixelPlot(percent_drift_per_year, vmin=vmin, 
                            vmax=vmax,cmap=plt.cm.coolwarm, ax=ax2,
                            annotate=True)

            ticks = np.arange(vmin, vmax + 0.1, 1)
            norm = BoundaryNorm(np.linspace(vmin, vmax, endpoint=True), plt.cm.plasma.N)
            cbar = plt.gcf().colorbar(ScalarMappable(norm=norm, cmap=plt.cm.coolwarm), 
                                        shrink=0.9, ax=ax2, location='right', 
                                        label='Pixel drift / percent per yr')

            cbar.set_ticks(ticks=ticks, labels=[f"${x:+.0f}$" for x in ticks])
            ax2.set_title(f"{self.site} - Mirror {self.tel}")

            centerRow = 35 / 3.0
            elevation, rows = [], []
            for row in range(22):

                elevation.append((row - centerRow + 1) * 1.5 * np.sqrt(3) / 2)
                this_row = get_row(row)
                rows.append(this_row[~np.isnan(this_row)])

            ax1.boxplot(rows, positions=elevation, vert=False,
                        showfliers=False, notch=True, widths=0.8)
            ax1.set_yticks([])
            ax1.axvline(0, ls='--', c='k', lw=0.5)
        
            plt.subplots_adjust(wspace=0)

        return percent_drift_per_year       


    def add_shift_information(self) -> None:
        try:
            dates = iter(self.df.groupby(self.df["datetime"].dt.date))
            self.df["shift_day"] = 1
            old, _ = next(dates)
            increment = 1

            while True:
                new, df = next(dates)
                days_diff = (new - old).days

                if days_diff == 1:
                    increment += 1

                elif days_diff < 5:
                    increment += days_diff

                else:
                    increment = 1

                self.df.loc[df.index, "shift_day"] = increment
                old = new

        except StopIteration: pass

class PixelNSB():

    def __init__(self, pixel, site, tel, df) -> None:
        self.pixel = pixel
        self.site = site
        self.tel = tel
        self.df = df.copy()


    def __call__(self, col="nigt_sky_background") -> (np.ndarray):
        return self.df["datetime"].values, self.df[col].values


    def show(self, **kwargs):

        show_mean = kwargs.get("show_mean", False)
        show_jump = kwargs.get("show_jump", False)
        show_drift = kwargs.get("show_drift", False)
        ax = kwargs.get("ax", plt.gca())

        opacity = 0.1 if show_mean or show_drift else 1

        x = self.df["datetime"]
        xt = self.df["days_since_start"]
        y = self.df["night_sky_background"]
        
        ax.scatter(x, y, 
                    alpha=opacity, 
                    ec='none')

        if show_mean:
            ax.plot(*tools.running_mean(x, y, 7 * 30), marker='none',
                    label="1-month rolling mean")

        if show_drift:
            popt, pcov = np.polyfit(xt, y, deg=1, cov=True)
            f = np.poly1d(popt)

            drift_params = uncertainties.correlated_values(popt, pcov)
            value = drift_params[0]/f(0) * 36500
            value = f"{value.nominal_value:.2f}\pm{value.std_dev:.2f}"

            ax.plot(x, f(xt), label=f"${value}$ %/yr", marker='none')

        if show_jump:
            result = self.get_jumps()
            ax.axvline(x.values[result[0]], linestyle="--", 
                       lw=0.3, color="orange")

        ax.set_ylabel("$\sigma^2_\mathrm{NSB}$ / ADC$^2$")
        ax.legend(title=f'{self.site} {self.tel} $-$ Pixel #{self.pixel}')
        plot.apply_datetime_format(ax)

    
    def get_jumps(self, n_bkps=1, col="night_sky_background") -> list:

        algo = rpt.Binseg(model="l1").fit(self.df[col].values)
        return algo.predict(n_bkps=n_bkps)

    
    def periodogram(self, show=False) -> Periodogram:

        periodogram = Periodogram(self.df["days_since_start"],
                                  self.df["night_sky_background"])

        if show:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            self.show(ax=ax1, show_mean=True)
            periodogram.show(ax2)

        self._periodogram = periodogram
        return periodogram


    def get_model(self, show=False, **peak_kwargs) -> callable:
        
        if not hasattr(self, "_periodogram"):
            _ = self.periodogram(show=False)

        n_peaks = peak_kwargs.pop('n_peaks', 1)
        period, power = self._periodogram.power_spectral_density()
        peaks = find_peaks(power, **peak_kwargs)[0][:-n_peaks-1:-1]

        def fct(x):
            y_model = np.zeros_like(x)
            
            for peak in peaks:
                y_model += self._periodogram.model(x, 1/period[peak]) - self._periodogram.offset()
            y_model += self._periodogram.offset()

            return y_model
        
        if show:
            _, (ax1, ax2) = plt.subplots(2, 1)
            self.show(ax=ax1, show_mean=True)
            
            for peak in peaks:
                ax2.axvline(period[peak], c='red', ls='--', lw=0.5)

            self._periodogram.show(ax2)
            ax1.plot(self.df["datetime"], fct(self.df["days_since_start"]), c='red', marker='none')

        return fct
    

    def add_cloud_information(self):

        m = {
            "LosLeones" : 0,
            "LosMorados" : 6,
            "LomaAmarilla" : 12,
            "Coihueco" : 18,
            "Heat" : 24
        }[self.site] + self.tel

        clouds_db = "/cr/tempdata01/filip/clouds_2010_2024/"
        times, mask = np.loadtxt(f"{clouds_db}/clouds_combined_m{m}.txt", 
                                dtype=int, usecols=[0, self.pixel], unpack=True)
        
        gps_to_datetime = lambda t: datetime.fromtimestamp(t + GPS_OFFSET)
        self.df['cloud_time_diff'] = np.nan
        self.df['clouds'] = 1
        for i in range(len(times)):

            time = gps_to_datetime(times[i])
            closest_df_row = np.argmin(np.abs(self.df['datetime'] - time))
            cloud_time_diff = np.abs((self.df.iloc[closest_df_row, 1] - time).total_seconds())

            if cloud_time_diff > 3600: continue

            self.df.iloc[closest_df_row, -2] = cloud_time_diff
            self.df.iloc[closest_df_row, -1] = mask[i]