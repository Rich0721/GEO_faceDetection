# GEO_faceDetection

<div>
<p><h2>File Explanation</h2></p>
	<div>
	<p>classification.py: 將facephoto的人臉部份切割出來，並分成train、veri、test三個資料夾，並放入dataset</p>
	<p>-s 指定原始路徑資料夾，預設"facephoto"</p>
	<p>-d 儲存資料夾，預設"dataset"</p>
	<p>-n 訓練檔案數量， 預設為200張</p>
	<p>-t 切割模式， 「1」: 戴口罩和沒戴口罩為同一類別
				「2」: 只使用沒戴口罩
			   「3」: 只使用戴口罩
				 「4」: 戴口罩和沒戴口罩為不同類別</p>
	<p>python classification.py -t 1
	</div>
	<div>
	<p>CNNtrainAndTest.py : 訓練檔案，訓練資料夾設定為"dataset/train"  驗證資料夾"dataset/veri"</p>
	<p>-load 載入以前訓練的權重檔，重新訓練時可以使用</p>
	<p>-storage 儲存權重檔</p>
	<p>上述檔案只能存成".h5"或".hdf5"</p>
	<p>python CNNtrainAndTest.py -load XXX.hdf5 -storage XXXX.hdf5</p>
	</div>
	<div>
	<p>executeFace_image.py: 指定影像檔辨識</p>
	<p>-image 圖片檔名稱</p>
	</div>
	<div>
	<p>executeFace_video.py: 影像的人臉辨識</p>
	<p>gui.py: 蒐集員工影像</p>
	<p>ProcessViedo: 將影像切割成圖片</p>
	</div>
</div>

