import unittest
import numpy as np
import warp
import alignment
import blend
import cv2

class TestWarp(unittest.TestCase):
    '''These TestCases tests the warping functions.'''
    @classmethod
    def setUp(self):
        ''' Run the warps once (independent of thresholds) '''
        #blank image
        blank = np.asarray(np.ones((40, 40, 3))*255.0, dtype=np.uint8)
        #simple grid image
        grid = np.asarray(np.ones((40, 40, 3))*255.0, dtype=np.uint8)
        grid[(10,30),:,:]=0
        grid[:,(10,30),:]=0
        parameters = (20,0.1,-0.1)

        resBl = np.load('testMat/warpBlank.npy')

        self.img_bl = warp.warpSpherical(blank,parameters[0],parameters[1],parameters[2])
        self.org_bl = resBl


    def test_computeSphericalWarpMappings(self):
        ''' Check if spherical warp is correct. '''
        self.assertTrue(np.allclose(self.img_bl, self.org_bl, rtol=1e-05, atol=1e-05),
            'Error in Spherical warping'
        )


class TestAlignment(unittest.TestCase):
    '''These TestCases tests the alignment functions.'''

    @classmethod
    def setUp(self):
        self.f1 = []
        self.f2 = []
        self.matches = []

        self.outlier_f1 = cv2.KeyPoint(2,2,4)
        self.outlier_f2 = cv2.KeyPoint(3,3,4)

        for i in range(4):
            feature = cv2.KeyPoint(i//2,i%2, 4)
            self.f1.append(feature)
            self.f2.append(feature)
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = i
            match.distance = 0
            self.matches.append(match)

        self.matches_with_outlier = self.matches[:]
        match = cv2.DMatch()
        match.queryIdx = 4
        match.trainIdx = 4
        match.distance = 0
        self.matches_with_outlier.append(match)

    def tearDown(self):
        pass

    def test_computehomography2(self):
        '''Tests A matrix from TODO 2'''
        # Place holder to get A from computeHomography
        A_student = np.zeros((8, 9))
        alignment.computeHomography(self.f1, self.f2, self.matches, A_student)
        A_soln = np.load('testMat/identityA.npy')
        self.assertTrue(np.allclose(A_soln, A_student, rtol=1e-05, atol=1e-05),
            'Error in Filling in A Matrix'
        )
    def test_computehomography3(self):
        '''Tests A matrix from TODO 3'''
        H_student = alignment.computeHomography(self.f1, self.f2, self.matches)
        H_student = H_student.astype(float)
        H_student = H_student/H_student[2,2]
        self.assertTrue(np.allclose(np.eye(3), H_student, rtol=1e-05, atol=1e-05),
            'Error in Computing Homography'
        )
    def test_alignPair(self):
        '''Tests TODO 4'''
        M = alignment.alignPair(self.f1,self.f2,self.matches, alignment.eHomography, 1, 1)

    def test_getInliers(self):
        '''Tests TODO 5'''
        inliers = alignment.getInliers(self.f1+[self.outlier_f1],self.f2+[self.outlier_f2],self.matches_with_outlier, np.eye(3),1)
        self.assertTrue(len(inliers)==4,"Error in getting inliers")
        inliers = alignment.getInliers(self.f1+[self.outlier_f1],self.f2+[self.outlier_f2],self.matches_with_outlier, np.eye(3),2)
        self.assertTrue(len(inliers)==5,"Error in getting inliers")

    def test_leastSquaresFit(self):
        '''Tests TODO 6,7'''
        M = alignment.leastSquaresFit(self.f1+[self.outlier_f1],self.f2+[self.outlier_f2],self.matches_with_outlier,1,[0,1,2,3])
        M = M.astype(float)
        M = M/M[2,2]
        self.assertTrue(np.allclose(np.eye(3), M, rtol=1e-05, atol=1e-05),
            'Error in least square fitting'
        )
        M = alignment.leastSquaresFit(self.f1+[self.outlier_f1],self.f2+[self.outlier_f2],self.matches_with_outlier,1,[0,1,2,4])
        M = M.astype(float)
        M = M/M[2,2]
        transform = np.array([[0.9,0,0],[0,0.9,0],[-0.1,-0.1,1]])
        self.assertTrue(np.allclose(transform, M, rtol=1e-05, atol=1e-05),
            'Error in least square fitting'
        )

class TestBlend(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.testimage = np.zeros((10,10,3))
        self.rot_trans_transform = np.array([[np.cos(np.pi/4),-np.sin(np.pi/4),5],[np.sin(np.pi/4),np.cos(np.pi/4),-5],[0,0,1]])
        self.rot_trans_transform1 = np.array([[1,0,-5],[0,1,5],[0,0,1]])
        self.rot_trans_transform2 = np.array([[1,0,5],[0,1,-5],[0,0,1]])

        self.acc = np.zeros((50,75,4))
        self.img1 = np.ones((50,50,3))
        self.img2 = np.full((50,50,3),2)
        self.transform = np.array([[1,0,25],[0,1,0],[0,0,1]])


    def test_imageBoundingBox(self):
        '''Tests TODO 8'''
        minX,minY,maxX,maxY = blend.imageBoundingBox(self.testimage,self.rot_trans_transform)
        sol_minX,sol_minY,sol_maxX,sol_maxY = \
            int(5-9*np.sin(np.pi/4)),int(-5),int(5+9*np.sin(np.pi/4)),int(18*np.sin(np.pi/4)-5)
        self.assertAlmostEqual(minX, sol_minX,
            msg='Expected bounding box min x to be {} +/-1 but got {}.'.format(sol_minX,
            minX),
            delta=1.01,
        )
        self.assertAlmostEqual(maxY, sol_maxY,
            msg='Expected bounding box max y to be {} +/-1 but got {}.'.format(sol_maxY,
            maxY),
            delta=1.01,
        )
        self.assertAlmostEqual(maxX, sol_maxX,
            msg='Expected bounding box max x to be {} +/-1 but got {}.'.format(sol_maxX,
            maxX),
            delta=1.01,
        )
        self.assertAlmostEqual(minY, sol_minY,
            msg='Expected bounding box min y to be {} +/-1 but got {}.'.format(sol_minY,
            minY),
            delta=1.01,
        )

    def test_getAccSize(self):
        '''Tests TODO 9'''
        ipv = [blend.ImageInfo("test1",self.testimage,self.rot_trans_transform1),
            blend.ImageInfo("test2",self.testimage,self.rot_trans_transform2)]
        accWidth, accHeight, channels, width, translation = blend.getAccSize(ipv)
        self.assertAlmostEqual(accWidth, 20,
            msg='Expected acc width to be {} +/-1 but got {}.'.format(20,
            accWidth),
            delta=1.01,
        )
        self.assertAlmostEqual(accHeight, 20,
            msg='Expected acc height to be {} +/-1 but got {}.'.format(20,
            accHeight),
            delta=1.01,
        )


if __name__ == '__main__':
    unittest.main()
