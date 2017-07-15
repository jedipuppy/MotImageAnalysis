□導入方法

Githubに最新版をあげている
https://github.com/jedipuppy/MotImageAnalysis
Github使いは適宜Cloneして使う。そうでない場合はdownloadして使う。

-環境構築
Python3.x,OpenCV3.xが必要。導入はanacondaを推奨している。
Windowsの場合はAnacondaの公式ウェブサイト（https://www.continuum.io/downloads）から最新版のAnacondaをダウンロード、インストールする。Linuxの場合は同じくウェブサイトからダウンロードして
bash Anaconda3-4.4.0-Linux-x86_64.sh 
を叩く


次にOpenCVを導入する。要注意なのが２０１７年７月時点でPython3.6がOpenCV3に対応できていない。そのためconda-forgeチャンネルを追加する。
conda config --add channels conda-forge
そして
conda install opencv
を叩く。途中いろいろときいてくるので適宜「y」をおす。管理者権限が必要なのでwindowsならcmdを管理者権限で立ち上げる必要があるし、linuxならsudoをつけるのを忘れないように。

これでOpenCVが入っていれば
python
と叩いてpythonを起動した後
import cv2
と呼び出してもエラーが出ないはず。

------------------------------------------------------------------------------------------------------
□使い方
run26ディレクトリ（201707xx-xxxxxxといったディレクトリがある場所）内にMotImageAnalysis.pyを置く。terminal等で
MotImageAnalysis.py filename over_threshold under_threshold vmax bg_img x y x2 y2
と打つ

-filename
201707xx-xxxxxxといったディレクトリ名をいれる

-over_threshold
MOTがあると考えられるintensity/pixelの閾値を決める。何度も走らせながら最適な値を探索するとよい

-under_threshold
MOTがないと考えられるintensity/pixelの閾値を決める。何度も走らせながら最適な値を探索するとよい

-vmax 
ヒートマップの上限値を決める。下限値は0に固定している

-bg_img 
背景画像（各画像から背景として差し引く画像）のファイル番号を決める。できるだけintensity/pixelが小さい番号がのぞましい。そうでないときは2d color map (threshold)等がMOTの部分だけ抜き出るような画像になる

-x,y,x2,y2
MOTの場所。Rbのテストから原則的に78 48 89 59に固定でいいと思われる。

------------------------------------------------------------------------------------------------------
□表示内容
- time evolution of intensity：青線各フレーム画像から背景画像を差し引き、ROI上での輝度平均をとったもの。赤はover thresholdで緑がunder threshold。赤が有意な青線のピークの少し下にくるようにした、緑線を青線のノイズに対して下側にくるように調整する

-cross-section image：赤線がover thresholdを超えた画像から背景画像を差し引いたものの平均についてROI中心のx断面図。緑はunder thresholdのそれ

-Fourier ：time evolution of intensityのフーリエ変換。周期的に中性化器を動かした場合等に有意な周波数成分を確認することができる

-2d color map(over threshold, under threhsold, difference)：それぞれover thresholdを超えた画像から背景画像をひいたものの平均、under thresholdのそれ、その二つの差をとったもの。differenceでMOT信号がみえるように調整する（vmax等）

-2d color map(over threshold, under threhsold, difference)：2d color mapのROI上のもの
