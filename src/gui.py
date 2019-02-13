import argparse
import os

import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.ttk as ttk

import cv2
import numpy as np

import alignment
import blend
import pyuiutils.uiutils as uiutils
import warp

DEFAULT_FOCAL_LENGTH = 678
DEFAULT_K1 = -0.21
DEFAULT_K2 = 0.26



def parse_args():
    parser = argparse.ArgumentParser(description="Panorama Maker")
    parser.add_argument(
        "--extra-credit", dest="ec", action='store_true',
        help="Flag to toggle extra credit features"
    )
    return parser.parse_args()


class AutostitchUIFrame(tk.Frame):
    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.notebook = ttk.Notebook(self.parent)
        self.notebook.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.notebook.add(
            HomographyVisualizationFrame(self.notebook, root),
            text='Homography'
        )
        self.notebook.add(
            SphericalWarpFrame(self.notebook, root), text='Spherical Warp'
        )
        self.notebook.add(
            AlignmentFrame(self.notebook, root), text='Alignment'
        )
        self.notebook.add(
            PanoramaFrame(self.notebook, root), text='Panorama'
        )
        self.notebook.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)

    def updateUI(self):
        self.parent.update()


class AutostitchBaseFrame(uiutils.BaseFrame):
    '''The base frame shared by all the tabs in the UI.'''

    def __init__(self, parent, root, nrows, ncolumns):
        assert nrows >= 2 and ncolumns >= 1
        uiutils.BaseFrame.__init__(
            self, parent, root, nrows, ncolumns,
            initial_status='Welcome to Autostitch UI'
        )

        self.imageCanvas = uiutils.ImageWidget(self)
        self.imageCanvas.grid(
            row=nrows - 2, columnspan=ncolumns,
            sticky=tk.N + tk.S + tk.E + tk.W,
        )

        self.grid_rowconfigure(nrows - 2, weight=1)

    def setImage(self, cvImage):
        if cvImage is not None:
            self.imageCanvas.draw_cv_image(cvImage)

    def saveScreenshot(self):
        if self.imageCanvas.has_image():
            filename = tkFileDialog.asksaveasfilename(
                parent=self, filetypes=uiutils.supportedFiletypes,
                defaultextension=".png"
            )
            if filename:
                self.imageCanvas.write_to_file(filename)
                self.set_status('Saved screenshot to ' + filename)
        else:
            uiutils.error('Load image before taking a screenshot!')


class HomographyVisualizationFrame(AutostitchBaseFrame):
    def __init__(self, parent, root):
        AutostitchBaseFrame.__init__(self, parent, root, 3, 3)

        tk.Button(self, text='Load Image', command=self.loadImage).grid(
            row=0, column=0, sticky=tk.W + tk.E)

        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=1, sticky=tk.W + tk.E)

        tk.Button(self, text='Apply Homography', command=self.applyHomography) \
            .grid(row=0, column=2, sticky=tk.W + tk.E)

        self.image = None

    def loadImage(self):
        filename, image = self.ask_for_image()
        if image is not None:
            self.image = image
            self.setImage(image)

    def applyHomography(self):
        if self.image is not None:
            homography = uiutils.showMatrixDialog(
                self, text='Apply', rows=3, columns=3
            )
            if homography is not None:
                height, width, _ = self.image.shape
                self.setImage(cv2.warpPerspective(
                    self.image, homography, (width, height))
                )
                self.set_status('Applied the homography.')
        else:
            uiutils.error('Select an image before applying a homography!')


class SphericalWarpFrame(AutostitchBaseFrame):
    def __init__(self, parent, root):
        AutostitchBaseFrame.__init__(self, parent, root, 7, 6)

        tk.Button(self, text='Load Image', command=self.loadImage).grid(
            row=0, column=0, columnspan=2, sticky=tk.W + tk.E)

        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=2, columnspan=2, sticky=tk.W + tk.E)

        tk.Button(self, text='Warp Image', command=self.warpImage) \
            .grid(row=0, column=4, columnspan=2, sticky=tk.W + tk.E)

        # TODO: specify units and correct ranges
        tk.Label(self, text='Focal Length').grid(
            row=1, column=0, columnspan=2, sticky=tk.W
        )
        self.focalLengthEntry = tk.Entry(self)
        self.focalLengthEntry.insert(0, str(DEFAULT_FOCAL_LENGTH))
        self.focalLengthEntry.grid(
            row=1, column=2, columnspan=2, sticky=tk.W+tk.E
        )

        tk.Label(self, text='k1:').grid(row=2, column=4, sticky=tk.W)
        self.k1Entry = tk.Entry(self)
        self.k1Entry.insert(0, str(DEFAULT_K1))
        self.k1Entry.grid(row=2, column=5, sticky=tk.W + tk.E)

        tk.Label(self, text='k2:').grid(row=3, column=4, sticky=tk.W)
        self.k2Entry = tk.Entry(self)
        self.k2Entry.insert(0, str(DEFAULT_K2))
        self.k2Entry.grid(row=3, column=5, sticky=tk.W + tk.E)

        self.image = None

    def loadImage(self):
        filename, image = self.ask_for_image()
        if image is not None:
            self.image = image
            self.setImage(image)

    def getK1(self):
        k1 = DEFAULT_K1
        try:
            k1 = float(self.k1Entry.get())
        except:
            uiutils.error('You entered an invalid k1! Please try again.')
        return k1

    def getK2(self):
        k2 = DEFAULT_K2
        try:
            k2 = float(self.k2Entry.get())
        except:
            uiutils.error('You entered an invalid k2! Please try again.')
        return k2

    def warpImage(self, *args):
        if self.image is not None:
            focalLength = float(self.focalLengthEntry.get())
            k1 = self.getK1()
            k2 = self.getK2()
            warpedImage = warp.warpSpherical(self.image, focalLength, k1, k2)
            self.setImage(warpedImage)
            self.set_status('Warped image with focal length ' + str(focalLength))
        elif len(args) == 0:  # i.e., click on the button
            uiutils.error('Select an image before warping!')


class StitchingBaseFrame(AutostitchBaseFrame):
    def __init__(self, parent, root, nrows, ncolumns):
        AutostitchBaseFrame.__init__(self, parent, root, nrows, ncolumns)

        self.motionModelVar = tk.IntVar()

        tk.Label(self, text='Motion Model:').grid(row=0, column=2, sticky=tk.W)

        tk.Radiobutton(
            self, text='Translation', variable=self.motionModelVar,
            value=alignment.eTranslate
        ).grid(row=0, column=3, sticky=tk.W)

        tk.Radiobutton(
            self, text='Homography', variable=self.motionModelVar,
            value=alignment.eHomography
        ).grid(row=0, column=3, sticky=tk.E)

        self.motionModelVar.set(alignment.eHomography)

        tk.Label(
            self, text='Percent Top Matches for Alignment:'
        ).grid(row=1, column=0, sticky=tk.W)

        self.matchPercentSlider = tk.Scale(
            self, from_=0.0, to=100.0, resolution=1, orient=tk.HORIZONTAL
        )
        self.matchPercentSlider.set(20.0)
        self.matchPercentSlider.grid(row=1, column=1, sticky=tk.W + tk.E)
        self.matchPercentSlider.bind("<ButtonRelease-1>", self.compute)

        tk.Label(self, text='Number of RANSAC Rounds:').grid(
            row=1, column=2, sticky=tk.W
        )

        # TODO: determine sane values for this
        self.nRANSACSlider = tk.Scale(
            self, from_=1, to=10000, resolution=10, orient=tk.HORIZONTAL
        )
        self.nRANSACSlider.set(500)
        self.nRANSACSlider.grid(row=1, column=3, sticky=tk.W + tk.E)
        self.nRANSACSlider.bind("<ButtonRelease-1>", self.compute)

        tk.Label(self, text='RANSAC Threshold:').grid(
            row=1, column=4, sticky=tk.W
        )

        # TODO: determine sane values for this
        self.RANSACThresholdSlider = tk.Scale(
            self, from_=0.1, to=100, resolution=0.1, orient=tk.HORIZONTAL
        )
        self.RANSACThresholdSlider.set(5)
        self.RANSACThresholdSlider.grid(row=1, column=5, sticky=tk.W + tk.E)
        self.RANSACThresholdSlider.bind("<ButtonRelease-1>", self.compute)

        tk.Label(self, text='Focal Length (pixels):').grid(
            row=2, column=4, sticky=tk.W
        )
        self.focalLengthEntry = tk.Entry(self)
        self.focalLengthEntry.insert(0, str(DEFAULT_FOCAL_LENGTH))
        self.focalLengthEntry.grid(row=2, column=5, sticky=tk.W + tk.E)

        tk.Label(self, text='k1:').grid(row=3, column=4, sticky=tk.W)
        self.k1Entry = tk.Entry(self)
        self.k1Entry.insert(0, str(DEFAULT_K1))
        self.k1Entry.grid(row=3, column=5, sticky=tk.W + tk.E)

        tk.Label(self, text='k2:').grid(row=4, column=4, sticky=tk.W)
        self.k2Entry = tk.Entry(self)
        self.k2Entry.insert(0, str(DEFAULT_K2))
        self.k2Entry.grid(row=4, column=5, sticky=tk.W + tk.E)

    def computeMapping(self, leftImage, rightImage):
        leftGrey = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
        rightGrey = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        leftKeypoints, leftDescriptors = orb.detectAndCompute(leftGrey, None)
        rightKeypoints, rightDescriptors = orb.detectAndCompute(rightGrey, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(leftDescriptors, rightDescriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        nMatches = int(
            float(self.matchPercentSlider.get()) * len(matches) / 100
        )

        if nMatches < 4:
            return None

        matches = matches[:nMatches]
        motionModel = self.motionModelVar.get()
        nRANSAC = int(self.nRANSACSlider.get())
        RANSACThreshold = float(self.RANSACThresholdSlider.get())

        return alignment.alignPair(
            leftKeypoints, rightKeypoints, matches, motionModel, nRANSAC,
            RANSACThreshold
        )

    def compute(self, *args):
        raise NotImplementedError('Implement the computation')

    def getFocalLength(self):
        focalLength = 0
        try:
            focalLength = float(self.focalLengthEntry.get())
            if focalLength > 0:
                return focalLength
        except:
            pass
        uiutils.error('You entered an invalid focal length! Please try again.')
        return 0

    def getK1(self):
        k1 = DEFAULT_K1
        try:
            k1 = float(self.k1Entry.get())
        except:
            uiutils.error('You entered an invalid k1! Please try again.')
        return k1

    def getK2(self):
        k2 = DEFAULT_K2
        try:
            k2 = float(self.k2Entry.get())
        except:
            uiutils.error('You entered an invalid k2! Please try again.')
        return k2


class AlignmentFrame(StitchingBaseFrame):
    def __init__(self, parent, root):
        StitchingBaseFrame.__init__(self, parent, root, 9, 6)

        tk.Button(self, text='Load Left Image', command=self.loadLeftImage)\
            .grid(row=0, column=0, sticky=tk.W + tk.E)

        tk.Button(self, text='Load Right Image', command=self.loadRightImage)\
            .grid(row=0, column=1, sticky=tk.W + tk.E)

        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=4, sticky=tk.W + tk.E)

        tk.Button(self, text='Align Images', command=self.alignImagesClick) \
            .grid(row=0, column=5, sticky=tk.W + tk.E)

        self.leftImage = None
        self.rightImage = None

    def loadLeftImage(self):
        filename, image = self.ask_for_image()
        if image is not None:
            self.leftImage = image
            self.applyVisualization()

    def loadRightImage(self):
        filename, image = self.ask_for_image()
        if image is not None:
            self.rightImage = image
            self.applyVisualization()

    def applyVisualization(self):
        self.setImage(uiutils.concatImages([self.leftImage, self.rightImage]))

    def alignImagesClick(self):
        if self.leftImage is None or self.rightImage is None:
            uiutils.error(
                'Both the images must be selected for alignment to '
                'be possible!'
            )
        else:
            self.compute()

    def compute(self, *args):
        if self.leftImage is not None and self.rightImage is not None:
            focalLength = self.getFocalLength()
            k1 = self.getK1()
            k2 = self.getK2()
            if focalLength <= 0:
                return
            if self.motionModelVar.get() == alignment.eTranslate:
                left = warp.warpSpherical(self.leftImage, focalLength, k1, k2)
                right = warp.warpSpherical(
                    self.rightImage, focalLength, k1, k2
                )
            else:
                left = self.leftImage
                right = self.rightImage
            mapping = self.computeMapping(left, right)
            height, width, _ = right.shape

            # TODO what if the mapping is singular?
            mapping = np.linalg.inv(mapping)
            mapping /= mapping[2, 2]

            points = np.array([
                [0, 0, 1],
                [width, 0, 1],
                [0, height, 1],
                [width, height, 1],
            ], dtype=float)
            trans_points = np.dot(mapping, points.T).T
            trans_points /= trans_points[:, 2][:, np.newaxis]

            all_points = np.vstack([points, trans_points])

            minX = np.min(all_points[:, 0])
            maxX = np.max(all_points[:, 0])
            minY = np.min(all_points[:, 1])
            maxY = np.max(all_points[:, 1])

            # Create an accumulator image
            newWidth = int(np.ceil(maxX) - np.floor(minX))
            newHeight = int(np.ceil(maxY) - np.floor(minY))

            translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

            warpedRightImage = cv2.warpPerspective(
                right, np.dot(translation, mapping), (newWidth, newHeight)
            )
            warpedLeftImage = cv2.warpPerspective(
                left, translation, (newWidth, newHeight)
            )

            alpha = 0.5
            beta = 1.0 - alpha
            gamma = 0.0
            dst = cv2.addWeighted(
                warpedLeftImage, alpha, warpedRightImage, beta, gamma
            )

            self.setImage(dst)


class PanoramaFrame(StitchingBaseFrame):
    def __init__(self, parent, root):
        StitchingBaseFrame.__init__(self, parent, root, 9, 6)

        tk.Button(self, text='Load Directory', command=self.loadImages) \
            .grid(row=0, column=0, sticky=tk.W + tk.E)

        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=4, sticky=tk.W + tk.E)

        tk.Button(self, text='Stitch', command=self.compute) \
            .grid(row=0, column=5, sticky=tk.W + tk.E)

        tk.Label(self, text='Blend Width (pixels):').grid(
            row=2, column=0, sticky=tk.W
        )
        self.blendWidthSlider = tk.Scale(
            self, from_=0, to=200, resolution=1, orient=tk.HORIZONTAL
        )
        self.blendWidthSlider.grid(row=2, column=1, sticky=tk.W + tk.E)
        self.blendWidthSlider.set(50)

        self.is360Var = tk.IntVar()
        tk.Checkbutton(
            self, text='360 degree Panorama?', variable=self.is360Var,
            offvalue=0, onvalue=1
        ).grid(row=2, column=3, sticky=tk.W)
        self.is360Var.set(0)

        self.images = None

    def loadImages(self):
        dirpath = tkFileDialog.askdirectory(parent=self)
        if not dirpath:
            return
        files = sorted(os.listdir(dirpath))
        files = [
            f for f in files
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.ppm')
        ]
        self.images = [cv2.imread(os.path.join(dirpath, i)) for i in files]
        self.setImage(uiutils.concatImages(self.images))
        self.set_status(
            'Loaded {0} images from {1}'.format(len(self.images), dirpath)
        )

    def getK1(self):
        k1 = DEFAULT_K1
        try:
            k1 = float(self.k1Entry.get())
        except:
            uiutils.error('You entered an invalid k1! Please try again.')
        return k1

    def getK2(self):
        k2 = DEFAULT_K2
        try:
            k2 = float(self.k2Entry.get())
        except:
            uiutils.error('You entered an invalid k2! Please try again.')
        return k2

    def compute(self, *args):
        if self.images is not None and len(self.images) > 0:
            f = self.getFocalLength()
            if f <= 0:
                return
            k1 = self.getK1()
            k2 = self.getK2()

            processedImages = None

            if self.motionModelVar.get() == alignment.eTranslate:
                processedImages = [
                    warp.warpSpherical(i, f, k1, k2)
                    for i in self.images
                ]
            else:
                processedImages = self.images

            t = np.eye(3)
            ipv = []
            for i in range(0, len(processedImages) - 1):
                self.set_status(
                    'Computing mapping from {0} to {1}'.format(i, i+1)
                )
                ipv.append(
                    blend.ImageInfo('', processedImages[i], np.linalg.inv(t))
                )
                t = self.computeMapping(
                    processedImages[i], processedImages[i+1]
                ).dot(t)

            ipv.append(blend.ImageInfo(
                '', processedImages[len(processedImages)-1], np.linalg.inv(t))
            )

            t = self.computeMapping(
                processedImages[len(processedImages)-1],
                processedImages[0]
            ).dot(t)

            if self.is360Var.get():
                ipv.append(blend.ImageInfo(
                    '', processedImages[0], np.linalg.inv(t))
                )

            self.set_status('Blending Images')
            self.setImage(blend.blendImages(
                ipv, int(self.blendWidthSlider.get()),
                self.is360Var.get() == 1
            ))
            self.set_status('Panorama generated')
        else:
            uiutils.error(
                'Select a folder with images before creating the panorama!'
            )


if __name__ == '__main__':
    args = parse_args()
    root = tk.Tk()
    app = AutostitchUIFrame(root, root)
    root.title('Cornell CS 4670 - Autostitch Project')
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry("%dx%d+0+0" % (w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.mainloop()
