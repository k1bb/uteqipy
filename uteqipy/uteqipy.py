import glob
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.transforms import Affine2D
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from scipy.ndimage.morphology import binary_closing, binary_fill_holes
from skimage.measure import regionprops_table

####################################################################
#                            __________                            #
#                           / うてきパイ ）                          #
#                           ‾‾‾‾‾‾‾‾‾‾‾                            #
#    Unified Tool for Enhanced                                     #
#          Quantification and Imaging of Precipitation dYnamics    #
####################################################################


class Factory:
    ############################################################
    # (((三三三三三三三三三三三三三三 初期化 三三三三三三三三三三三三三三三))) #
    ############################################################
    def __init__(self, factory_name, factory_dir=None, dx=0.05):
        # ファクトリの名前を指定する
        self.factory_name = factory_name
        # ファクトリのファイルを保存するディレクトリ名を指定する
        if factory_dir is None:
            self.factory_dir = f"./{factory_name}"
        else:
            self.factory_dir = factory_dir
        # ファクトリのファイルを保存するディレクトリを作成する
        if not os.path.exists(self.factory_dir):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(self.factory_dir)
            print(f"Directory {self.factory_dir} created.")
        else:
            print(f"Directory {self.factory_dir} already exists.")
        # ファクトリファイルのパス
        self.original_file = f"{self.factory_dir}/{self.factory_name}_original.nc"
        self.cleaned_file = f"{self.factory_dir}/{self.factory_name}_cleaned.nc"
        self.blurred_file = f"{self.factory_dir}/{self.factory_name}_blurred.nc"
        self.binarized_file = f"{self.factory_dir}/{self.factory_name}_binarized.nc"
        self.labeled_file = f"{self.factory_dir}/{self.factory_name}_labeled.nc"
        self.fitted_file = f"{self.factory_dir}/{self.factory_name}_fitted.csv"
        func = {
            self.original_file: "read_frames",
            self.cleaned_file: "clean_frames",
            self.binarized_file: "binarize_frames",
            self.labeled_file: "label_frames",
            self.fitted_file: "fit_objects",
        }
        for file in [
            self.original_file,
            self.cleaned_file,
            self.binarized_file,
            self.labeled_file,
            self.fitted_file,
        ]:
            if os.path.exists(file):
                print(f"{file} already exists. You can skip {func[file]}.")
            else:
                print(f"{file} does not exist. Execute {func[file]}.")
        # ピクセル幅を設定する
        self.dx = dx
        # 並列処理に使用可能なCPU数を取得する
        self.max_workers = cpu_count()
        return

    ############################################################
    # (((三三三三三三三三三三 netcdfファイルを開く 三三三三三三三三三三三))) #
    ############################################################
    def _open_nc(self, path, chunks=None):
        return xr.open_dataarray(path, chunks=chunks)

    def original(self, chunks=None):
        return self._open_nc(self.original_file, chunks=chunks)

    def cleaned(self, chunks=None):
        return self._open_nc(self.cleaned_file, chunks=chunks)

    def blurred(self, chunks=None):
        return self._open_nc(self.blurred_file, chunks=chunks)

    def binarized(self, chunks=None):
        return self._open_nc(self.binarized_file, chunks=chunks)

    def labeled(self, chunks=None):
        return self._open_nc(self.labeled_file, chunks=chunks)

    ############################################################
    # (((三三三三三三三三三三三 csvファイルを開く 三三三三三三三三三三三三))) #
    ############################################################
    def _open_csv(self, path):
        return pd.read_csv(path, index_col=0, parse_dates=[2])

    def fitted(self):
        fitted = self._open_csv(self.fitted_file)
        fitted["time"] = pd.to_datetime(fitted["time"])
        return fitted

    ############################################################
    # (((三三三三三三三三三三三三 画像の読み込み 三三三三三三三三三三三三三))) #
    ############################################################
    def read_frames(self, input):
        # パスの一覧
        paths = sorted(glob.glob(input))
        # 並列処理で画像を読み込みDataArrayに変換する
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            frames = list(executor.map(_image_to_DataArray, paths))
        # 時刻情報を元にDataArrayを連結する
        original = xr.concat(frames, dim="time")
        # x, y軸を設定する
        x = self.dx * np.arange(len(original["x"]))
        y = self.dx * np.arange(len(original["y"]))
        original = original.assign_coords({"x": ("x", x), "y": ("y", y)})
        # 保存
        original.to_netcdf(self.original_file)
        return

    ############################################################
    # (((三三三三三三三三三三三三 オフセットの補正 三三三三三三三三三三三三))) #
    ############################################################
    def clean_frames(self, median_window_size=5):
        if median_window_size % 2 == 0:
            raise ValueError("median_window_size must be odd.")
        else:
            self.median_window_size = median_window_size
        # 元データを読み込む
        original = self.original(chunks={"time": "auto"}).transpose("y", "x", "time")
        # 並列処理で画像をラベル化する
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            cleaned = list(executor.map(self._clean_pixels, original))
        # 空間情報を元にDataArrayを連結する
        cleaned = xr.concat(cleaned, dim="y").transpose("time", "y", "x")
        # 保存
        cleaned.to_netcdf(self.cleaned_file)
        return

    # 一部のピクセルのオフセットを補正する
    def _clean_pixels(self, pixels):
        # 移動窓の中央値（＝背景）
        median_before = pixels.rolling(time=self.median_window_size).median().fillna(0)
        median_after = median_before.shift(time=-self.median_window_size + 1).fillna(0)
        # 背景の明度を差し引く
        dep_from_before = pixels - median_before
        dep_from_after = pixels - median_after
        # より暗い方を採用する
        cleaned = dep_from_before.where(
            dep_from_before < dep_from_after, dep_from_after
        )
        # 負のピクセルをゼロにする
        cleaned = cleaned.where(cleaned >= 0, 0)
        # 型をuint8に戻す
        cleaned = cleaned.astype("uint8")
        # 実際に計算する
        cleaned = cleaned.compute()
        return cleaned

    ############################################################
    # (((三三三三三三三三三三三三三三 二値化 三三三三三三三三三三三三三三三))) #
    ############################################################
    def binarize_frames(self, gaussian_sigma=1, binarize_threshold=3):
        cleaned = self.cleaned()
        self.gaussian_sigma = gaussian_sigma
        self.binarize_threshold = binarize_threshold
        # ノイズを抑制するためにガウシアンフィルタでぼかす
        # 並列処理で画像を2値化する
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            binarized = list(executor.map(self._binarize_a_frame, cleaned))
        # 時刻情報を元にDataArrayを連結する
        binarized = [b for b in binarized if b is not None]
        binarized = xr.concat(binarized, dim="time")
        # データ型を戻す
        binarized = binarized.astype("uint8")
        # 保存
        binarized.to_netcdf(self.binarized_file)
        return

    # Gaussianフィルタをかけて2値化する
    def _binarize_a_frame(self, frame):
        time = frame["time"].values
        # ガウシアンフィルタをかける
        blurred = xr.DataArray(
            name="brightness",
            data=gaussian_filter(frame.to_numpy(), self.gaussian_sigma),
            coords=frame.coords,
        ).expand_dims(time=[time])
        # 2値化
        binarized = xr.where(blurred > self.binarize_threshold, 255, 0).rename("binary")
        # 何か写っていたら結果を返し、何も写っていなければ終了
        if binarized.max(dim=["x", "y"]) == 255:
            return binarized
        else:
            return binarized

    ############################################################
    # (((三三三三三三三三三三三三三 ラベル付け 三三三三三三三三三三三三三三))) #
    ############################################################
    def label_frames(self, closing_mask_size=9):
        binarized = self.binarized()
        # 並列処理用にフレームをリスト化する
        bfs = [binarized.sel(time=time) for time in binarized["time"]]
        # closingの範囲をつくる
        self.closing_mask = _generate_closing_mask(closing_mask_size).to_numpy()
        # 並列処理で画像をラベル化する
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            labeled = list(executor.map(self._label_objects_in_a_frame, bfs))
        # 時刻情報を元にDataArrayを連結する
        labeled = xr.concat(labeled, dim="time")
        # 保存
        labeled.to_netcdf(self.labeled_file)
        return

    # 2値化されたフレーム中のオブジェクトをラベリングする
    def _label_objects_in_a_frame(self, binarized_frame):
        # 近接したピクセルをつなげる
        closed = binary_closing(binarized_frame, self.closing_mask)
        # 穴を塞ぐ
        filled = binary_fill_holes(closed)
        # オブジェクトをラベリングする
        labeled = label(filled)[0]
        # DataArray化
        time = binarized_frame["time"].values
        labeled = (
            xr.DataArray(name="label", data=labeled, coords=binarized_frame.coords)
            .expand_dims(time=[time])
            .astype("uint8")
        )
        return labeled

    ############################################################
    # (((三三三三三三三三三三三三三 楕円近似 三三三三三三三三三三三三三三三))) #
    ############################################################
    def fit_objects(self):
        labeled = self.labeled()
        # 並列処理で画像をラベル化する
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            fitted = list(executor.map(self._fit_objects_in_a_frame, labeled))
        fitted = pd.concat(fitted)
        # 保存
        fitted.to_csv(self.fitted_file)
        return

    # ラベル付けされたフレーム中のオブジェクトを楕円近似する
    def _fit_objects_in_a_frame(self, labeled_frame):
        # 取得する要素
        properties = (
            "label",
            "axis_major_length",
            "axis_minor_length",
            "bbox",
            "centroid",
            "moments_normalized",
            "orientation",
        )
        # 測定
        props = pd.DataFrame(
            regionprops_table(labeled_frame.to_numpy(), properties=properties)
        )
        # 名前を変更する
        props = props.rename(
            columns={
                "axis_major_length": "width",
                "axis_minor_length": "height",
                "bbox-0": "bbox_bottom",
                "bbox-1": "bbox_left",
                "bbox-2": "bbox_top",
                "bbox-3": "bbox_right",
                "centroid-0": "y",
                "centroid-1": "x",
                "moments_normalized-0-2": "m02",
                "moments_normalized-1-1": "m11",
                "moments_normalized-2-0": "m20",
                "moments_normalized-0-3": "m03",
                "moments_normalized-1-2": "m12",
                "moments_normalized-2-1": "m21",
                "moments_normalized-3-0": "m30",
            }
        )
        # 短軸が1ピクセル以下の要素を削除
        props = props[props["height"] > 1]
        if len(props) == 0:
            return
        # 長さを合わせる
        props[
            ["x", "y", "bbox_left", "bbox_bottom", "bbox_right", "bbox_top", "width", "height"]
        ] = (
            self.dx
            * props[
                [
                    "x",
                    "y",
                    "bbox_left",
                    "bbox_bottom",
                    "bbox_right",
                    "bbox_top",
                    "width",
                    "height",
                ]
            ]
        )
        # 角度をラジアンから度に変換する
        props["angle"] = props["orientation"] - (np.pi / 2) * props[
            "orientation"
        ] / np.abs(props["orientation"])
        props["angle_deg"] = np.rad2deg(props["angle"])
        props["orientation"] = props["angle"] + np.pi / 2
        # アスペクト比を計算する
        props["aspect_ratio"] = props["height"] / props["width"]
        # 体積を計算する
        props["vol_ellipsoid"] = 4.0 * np.pi * (props["width"] / 2.0) ** 2 * (props["height"] / 2.0) / 3.0
        # 歪度を計算する
        sin = np.sin(props["orientation"])
        cos = np.cos(props["orientation"])
        m30 = props["m30"]
        m21 = props["m21"]
        m12 = props["m12"]
        m03 = props["m03"]
        m20 = props["m20"]
        m11 = props["m11"]
        m02 = props["m02"]
        props["skew"] = (
            -m30 * sin**3
            + 3 * m21 * sin**2 * cos
            - 3 * m12 * sin * cos**2
            + m03 * cos**3
        ) / (m20 * sin**2 - 2 * m11 * sin * cos + m02 * cos**2) ** (3 / 2)
        # bboxの高さ・幅
        props["bbox_height"] = props["bbox_top"] - props["bbox_bottom"]
        props["bbox_width"] = props["bbox_right"] - props["bbox_left"]
        # 時間の情報を追加
        props.insert(loc=0, column="time", value=labeled_frame["time"].to_pandas())
        # ファイル名を追加
        props.insert(loc=0, column="filename", value=labeled_frame["filename"].values)
        # indexを設定
        props["index"] = [
            f"{self.factory_name}_{labeled_frame['filename'].values}_{n:02}"
            for n in np.arange(len(props))
        ]
        props = props.set_index("index")
        # x, yの範囲を調べる
        xmax = self.dx * len(self.original()["x"])
        ymax = self.dx * len(self.original()["y"])
        # オブジェクトが細長過ぎたらフラグ
        props["too_elongated_flag"] = np.where(
            props["aspect_ratio"] > 2, True, np.where(props["aspect_ratio"] < 0.5, True, False)
        )
        # オブジェクトが境界に重なっていたらフラグ
        props["border_flag"] = [
            _check_border(row, xmax, ymax) for _, row in props.iterrows()
        ]
        # 2つ以上のオブジェクトが写っていたらフラグ
        props["multi_flag"] = len(props) >= 2
        # 並べ替え
        props = props[
            [
                "filename",
                "time",
                "x",
                "y",
                "angle",
                "angle_deg",
                "width",
                "height",
                "aspect_ratio", 
                "vol_ellipsoid",
                "skew",
                "bbox_bottom",
                "bbox_left",
                "bbox_top",
                "bbox_right",
                "bbox_width", 
                "bbox_height", 
                "multi_flag",
                "border_flag",
                "too_elongated_flag",
            ]
        ]
        return props

    ############################################################
    # (((三三三三三三三三三三三三三 結果表示 三三三三三三三三三三三三三三三))) #
    ############################################################
    def generate_report(self, im, show=False, save=True):
        fitted = self.fitted()
        time = im["time"].values
        objects_in_the_frame = fitted[fitted["time"] == time]

        xmax = self.dx * len(im["x"])
        ymax = self.dx * len(im["y"])

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        ax.set_title(f"{str(time)[:-6]} ({len(objects_in_the_frame)})", y=1.04)
        ax.tick_params(
            axis="both", which="both", top=True, bottom=True, right=True, left=True
        )
        ax.set_xticks(np.arange(0, xmax + 5, 5))
        ax.set_xticks(np.arange(0, xmax + 1, 1), minor=True)
        ax.set_xlim(xmax, 0)
        ax.set_yticks(np.arange(0, ymax + 5, 5))
        ax.set_yticks(np.arange(0, ymax + 1, 1), minor=True)
        ax.set_ylim(ymax, 0)
        ax.pcolormesh(im["x"], im["y"], im, cmap="binary_r", norm=Normalize(15, 50))
        if len(objects_in_the_frame) > 0:
            i = 0
            for index, obj in objects_in_the_frame.iterrows():
                c = cm.hsv(i / len(objects_in_the_frame))
                bottom_left = (
                    obj["x"] - obj["width"] / 2,
                    obj["y"] - obj["height"] / 2,
                )
                if obj["border_flag"]:
                    label = f"{index} (BORDER)"
                elif obj["too_elongated_flag"]:
                    label = f"{index} (TOO ELONGATED)"
                else:
                    label = f"{index} ({obj['width']:.2f}, {obj['height']:.2f}, {obj['aspect_ratio']:.2f}, {obj['angle']:.2f})"  # 角度の表記に注意!!!
                bbox = patches.Rectangle(
                    (obj["bbox_left"], obj["bbox_bottom"]),
                    width=obj["bbox_width"],
                    height=obj["bbox_height"],
                    edgecolor="w",
                    facecolor="none",
                    lw=0.5,
                )
                ax.add_patch(bbox)
                rectangle = patches.Rectangle(
                    bottom_left,
                    width=obj["width"],
                    height=obj["height"],
                    edgecolor=c,
                    facecolor="none",
                    lw=0.5,
                    label=label,
                    linestyle="dashed",
                )
                t = (
                    Affine2D().rotate_deg_around(obj["x"], obj["y"], -obj["angle"])
                    + ax.transData
                )
                rectangle.set_transform(t)
                ax.add_patch(rectangle)
                ax.text(
                    obj["x"] - obj["width"] * 0.75,
                    obj["y"],
                    index[-2:],
                    color=c,
                    fontsize=5,
                )
                i += 1
            fig.legend(fontsize=7, bbox_to_anchor=(0.5, 0), loc="upper center")
        else:
            ax.text(
                xmax / 2,
                ymax / 2,
                "NO DROPS",
                color="red",
                fontweight="bold",
                fontsize=30,
                ha="center",
                va="center",
            )
        if len(objects_in_the_frame) >= 2:
            ax.text(
                xmax / 2,
                ymax / 2,
                "MULTIPLE\nDROPS",
                color="yellow",
                fontweight="bold",
                fontsize=40,
                ha="center",
                va="center",
                alpha=0.5,
            )
        if save:
            # ディレクトリがなければ作成する
            if not os.path.exists(f"{self.factory_dir}/reports"):
                # ディレクトリが存在しない場合、ディレクトリを作成する
                os.makedirs(f"{self.factory_dir}/reports")
            # 保存
            plt.savefig(
                f"{self.factory_dir}/reports/{self.factory_name}_{im['filename'].values}.png",
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close()
        return

    # オブジェクトが写っているすべてのフレームのレポートを出力する
    def generate_all_reports(self):
        original = self.original()
        nframes = len(original["time"])
        # 並列処理で画像をラベル化する
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            _ = executor.map(self.generate_report, original)
        # エラーチェック
        # for r in _:
        #     print(r)
        return


# 画像データをDataArrayとして読み込む
def _image_to_DataArray(path):
    name, time = _path_to_time(path)
    data = np.array(Image.open(path))
    da = xr.DataArray(name="brightness", data=data, dims=["y", "x"]).expand_dims(
        time=[time]
    )
    # ファイル名の情報を追加
    da = da.assign_coords(filename=("time", [name]))
    return da


# ファイル名を時刻に変換
def _path_to_time(path):
    # 拡張子付きのファイル名
    name_with_extension = os.path.basename(path)
    # 拡張子を除いたファイル名
    name, _ = os.path.splitext(name_with_extension)
    year = name[0:4]
    month = name[4:6]
    date = name[6:8]
    hour = name[8:10]
    minute = name[10:12]
    second = name[12:14]
    millisecond = name[14:17]
    time = np.datetime64(
        f"{year}-{month}-{date}T{hour}:{minute}:{second}.{millisecond}"
    ).astype("datetime64[ns]")
    return name, time


# closingの範囲をつくる
def _generate_closing_mask(size):
    if size % 2 == 0:
        raise ValueError("Mask size must be odd.")
    mask_coord = np.arange(-(size - 1) / 2, (size + 1) / 2).astype("int")
    # 空のマスク
    mask = xr.DataArray(
        data=np.zeros([size, size]),
        coords={"i": mask_coord, "j": mask_coord},
    )
    # 中心からの距離
    r = np.sqrt(mask["i"] ** 2 + mask["j"] ** 2)
    mask = xr.where(r <= 0.5 * size, 1, 0).astype("int")
    return mask


# オブジェクトが境界に重なっているか調べる
def _check_border(row, xmax, ymax):
    if row["x"] < row["bbox_height"]:
        result = True
    elif row["x"] > xmax - row["bbox_height"]:
        result = True
    elif row["y"] < row["bbox_width"]:
        result = True
    elif row["y"] > ymax - row["bbox_width"]:
        result = True
    else:
        result = False
    return result
