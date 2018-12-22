import AudioFile
import Onsets
import SpectralFlux
import numpy as np
import soundfile as sf
import csv
feature = []


wavData = AudioFile.open("dataset_audio/14.wav")
q = sf.SoundFile('dataset_audio/14.wav')
print('samples = {}'.format(len(q)))
print('sample rate = {}'.format(q.samplerate))
print('seconds = {}'.format(len(q) / q.samplerate))

#mp3Data = AudioFile.open("test-stereo.mp3")

#fixedFrames = wavData.frames(753)

windowFunction = np.hamming
fixedFrames = AudioFile.frames(wavData,882,windowFunction)


#energyOnsets = Onsets.onsetsByEnergy(wavData)
#framesFromOnsets = wavData.framesFromOnsets(energyOnsets)

#print(fixedFrames[0].cqt()) 						# Constant Q Transform
#print(fixedFrames[0].dct())						# Discrete Cosine Transform
#print(np.sum(fixedFrames[0].energy(windowSize = 256))) 	# Energy
# fixedFrames[0].play()                       # Playback using pyAudio
#print(np.size(fixedFrames))
#fixedFrames[0].plot()                       # Plot using matplotlib
#print(fixedFrames[0].rms()) 						# Root-mean-squared amplitude
#print(fixedFrames[0].zcr()) 	

# Compute the spectra of each frame
i = 0
spectra = [f.spectrum() for f in fixedFrames]
flux = SpectralFlux.spectralFlux(spectra, rectify = True)
with open('data_set14.csv','w') as csvFile:

	for f in fixedFrames:
		
		spectra = f.spectrum() 
	
		MFCC_app = spectra.mfcc2()
		for j in range(13):
			feature.append(MFCC_app[j])

		feature.append(f.zcr())
		feature.append(np.sum(f.energy(windowSize=256)))
		feature.append(spectra.centroid())
		feature.append(spectra.rolloff())
		feature.append(flux[i])
		i = i + 1
		writer = csv.writer(csvFile)
		writer.writerow(feature)
		feature = [] 
csvFile.close()	 	


	#print(spectra.centroid()) 						# Spectral Centroid
	#print(spectra[0].chroma())							# Chroma vector
	#print(spectra[0].crest())                          # Spectral Crest Factor
	#spectra[0].flatness()                       # Spectral Flatness
	#print(spectra[0].idct())							# Inverse DCT
	#print(spectra[0].ifft())							# Inverse FFT
	#print(spectra[0].kurtosis())                       # Spectral Kurtosis
	#print(spectra[0].mean())                           # Spectral Mean
	#print('MFCC size:', np.size(spectra[0].mfcc2()))
	#print((spectra.mfcc2()))                          # MFCC (vectorized implementation)
	#print(spectra.plot())                           # Plot using matplotlib
	#print(spectra.rolloff())                        # Spectral Rolloff
	#print(spectra[0].skewness())                       # Spectral Skewness
	#print(spectra[0].spread())                         # Spectral Spread
	#print(spectra[0].variance())                       # Spectral Variance

	#flux = SpectralFlux.spectralFlux(spectra, rectify = True)
	#wavData.play()
	#print(flux)
#fixedFrames[0].play()