import numpy as np
import cv2

class fastglcm_wrapper:
    '''
    This function is the same as the fast_glcm code found in https://github.com/tzm030329/GLCM, it just loops around for multiple channels.
    Additionally I'm dividing each of the descriptive statistics by levels**2 to normalise the values
    
    Args:
        img (numpy.ndarray): The input image (h x w x ch), a 3D array where the third dimension is the number of channels.
        levels (int): The number of gray levels (quantization) to use when calculating the GLCM.
        kernel_size (int): The size of the kernel to use when filtering the GLCM.
        distance_offset (int): The pixel pair distance offset to use when calculating the GLCM.
        angles (list): A list of angles to use when calculating the GLCM.

    Example usage:
        tex2 = fastglcm_wrapper(im,levels=8,kernel_size=5,distance_offset=5,angles=[0,45,90,135])
        plot_rgb(tex2.calculate_glcm_entropy(),rb=2,gb=1,bb=0) # plot the entropy image
    '''

    def __init__(self, img, levels, kernel_size, distance_offset, angles) -> None:
        self.img = img
        self.h, self.w, self.k = img.shape
        self.levels = levels
        self.kernel_size = kernel_size
        self.distance_offset = distance_offset
        self.angles = angles
        self.glcm=self.calculate_glcm()

    def digitize(self):
        h, w, k = self.h, self.w, self.k
        digitized_channels = []

        for i in range(k):
            bins = np.linspace(np.min(self.img[..., i]), np.max(self.img[..., i]), self.levels + 1)
            digitized_channels.append(np.digitize(self.img[..., i], bins) - 1)

        digitized_image = np.stack(digitized_channels, axis=-1)
        return digitized_image
    
    def fast_glcm_modified(self,gl1,angle):
        '''Code taken from fastglcm except direct input the digitized image for one channel'''

        ks = self.kernel_size
        h,w = gl1.shape
        distance = self.distance_offset
        levels= self.levels

        # make shifted image
        dx = distance*np.cos(np.deg2rad(angle))
        dy = distance*np.sin(np.deg2rad(-angle))
        mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
        gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REPLICATE)

        # make glcm
        glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
        for i in range(levels):
            for j in range(levels):
                mask = ((gl1==i) & (gl2==j))
                glcm[i,j, mask] = 1

        kernel = np.ones((ks, ks), dtype=np.uint8)
        for i in range(levels):
            for j in range(levels):
                glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

        glcm = glcm.astype(np.float32)
        return glcm

    def calculate_glcm(self):
        k = self.k
        glcm_list = []
        gl = self.digitize()

        # looping over each channels and angles
        glcm_channels=[]
        for i in range(k):
            glcm_angles=[]
            for j in range(len(self.angles)):
                glcm_angles.append(self.fast_glcm_modified(gl[...,i],angle=self.angles[j]))

            glcm_channels.append(np.mean(np.stack(glcm_angles,axis=-1),axis=-1))
        return np.stack(glcm_channels,axis=-1)
    
    def calculate_glcm_mean(self):
        '''calculate mean'''
        levels=self.levels
        h,w,k = self.h,self.w,self.k
        glcm = self.glcm
        mean = np.zeros((h,w,k), dtype=np.float32)
        for i in range(1,levels):
            for j in range(levels):
                mean += glcm[i,j] * i 
        return mean/ (levels)**2
    
    def calculate_glcm_var(self):
        '''calc glcm var'''
        levels=self.levels
        glcm = self.glcm
        h,w,k = self.h,self.w,self.k
        mean = self.calculate_glcm_mean()

        var = np.zeros((h,w,k), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                var += glcm[i,j] * (i - mean)**2

        return var/ (levels)**4
    
    def calculate_glcm_contrast(self):
        levels=self.levels
        glcm = self.glcm
        h,w,k = self.h,self.w,self.k
        cont = np.zeros((h,w,k), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                cont += glcm[i,j] * (i-j)**2
        return cont/levels**4
    
    def calculate_glcm_dissimilarity(self):
        levels=self.levels
        glcm = self.glcm
        h,w,k = self.h,self.w,self.k
        diss = np.zeros((h,w,k), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                diss += glcm[i,j] * np.abs(i-j)
        return diss/levels**2
    
    def calculate_glcm_homogenity(self):
        levels=self.levels
        glcm = self.glcm
        h,w,k = self.h,self.w,self.k
        homo = np.zeros((h,w,k), dtype=np.float32)
        for i in range(1,levels):
            for j in range(levels):
                homo += i*glcm[i,j] / (1.+(i-j)**2)
        return homo/levels**2
    
    def calculate_glcm_asm(self):
        '''calc glcm angular second moment'''
        levels=self.levels
        glcm = self.glcm
        h,w,k = self.h,self.w,self.k
        asm = np.zeros((h,w,k), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                asm  += glcm[i,j]**2
        return asm/levels**2
    
    def calculate_glcm_entropy(self):
        levels = self.levels
        ks = self.kernel_size
        glcm = self.glcm
        pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
        ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
        return ent/levels**2
    
    def calculate_glcm_correlation(self):
        levels = self.levels
        glcm = self.glcm
        h,w,k = self.h,self.w,self.k
        corr = np.zeros((h,w,k), dtype=np.float32)
        
        mean_x = np.mean(glcm.sum(0),axis=0)
        mean_y = np.mean(glcm.sum(1),axis=0)

        std_x = np.std(glcm.sum(0),axis=0)
        std_y = np.std(glcm.sum(1),axis=0)

        for i in range(levels):
            for j in range(levels):
                corr += (i*j*glcm[i,j] - mean_x*mean_y)/(std_x*std_y)
        return corr/levels**2