"""
__File name__:          utils.py
__Author__:             Reto Furrer
__Date created__:       02.09.2019
__Date last modified__: 06.09.2019
__Python Version__:     3.6
__Project__:            IGT-Lab 2019 (02.09.2019-13.09.2019)
__Description__:        This file contains the functions used in the file IGTLab_RetoFurrer_2019 to 
                        analyse the topology of polymer tubes with microCT images. The functions might be 
                        called from any python IDE if needed.
"""

def create_directories():
    """
    Description:    Check whether the needed directories are present.  
    ------------    If not directories are created.   
    
    Inputs:
    -------
    none          
    
    Outputs:
    --------
    none
    """
    # Import packages needed for the function
    import os
    
    # Create directories in order to store exports and results propery if they not already exist
    ## Results (res)
    if not os.path.exists('res'):
        os.mkdir('res')
    
    ## Exports (res/diag)
    if not os.path.exists('res/diag'):
        os.mkdir('res/diag')
       
    ## Exports (res/txt)
    if not os.path.exists('res/txt'):
        os.mkdir('res/txt')
        
    ## Images (img)
    if not os.path.exists('img'):
        os.mkdir('img')

    ## Exports (exp)
    if not os.path.exists('exp'):
        os.mkdir('exp')
        
        
def print_original_image(image, sample="n/a", show="False", index=0):
    """
    Description:    Print and save original image with axis description 
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input image that need to be filtered 
    
    sample:         string
                    Defining which sample was choosen (default:"n/a")
                    
    show:           bool
                    Defines whether or not the result is plotted (default = "False")
                    
    index:          int
                    Index of the iteration to label the images
    
    Outputs:
    --------
    none
    """
    # Import packages needed for the function
    import matplotlib.pyplot as plt
    import os
    
    # Plot filtered image
    if show == "True":
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.title('Original image (sample: {}, index: {})'.format(sample,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')
        plt.show()
    
        # save it to a *.pdf file
        if not os.path.exists('exp/original'):
            os.mkdir('exp/original')
            plt.savefig('exp/original/img_raw_original{}_{}.pdf'.format(sample,index))
        else:    
            plt.savefig('exp/original/img_raw_original{}_{}.pdf'.format(sample,index))
    

def gaussian_filter(image, sigma, sample="n/a", show="False", index=0):
    """
    Description:    Applies a gaussian filter to an imagee (array) and returns the filtered image 
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input image that need to be filtered 
    
    sigma:          float
                    Defines the standard deviation of the gaussian filter

    sample:         string
                    Defining which sample was choosen (default:"n/a")
                    
    show:           bool
                    Defines whether or not the result is plotted (default:"False")
    
    index:          int
                    Index of the iteration to label the images (default:0)
    
    Outputs:
    --------
    img_filtered:   array
                    The filtered image
    """
    # Import packages needed for the function
    import matplotlib.pyplot as plt
    import scipy.ndimage as ndimage
    import os
    
    # Apply Gaussian filter to the gray image to reduce noise and show it to the user
    img_filtered = ndimage.gaussian_filter(image, sigma)
    
    # Plot filtered image
    if show == "True":
        plt.imshow(img_filtered, cmap='gray', interpolation='nearest')
        plt.title('filtered image, Gaussian (sample:{}, sigma: {}, index: {})'.format(sample,sigma,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')
        plt.show()
    
        # save it to a *.pdf file
        if not os.path.exists('exp/filt'):
            os.mkdir('exp/filt')
        if not os.path.exists('exp/filt/gauss'):
            os.mkdir('exp/filt/gauss')
            plt.savefig('exp/filt/gauss/img_filt_gaussian_sigma{}_sigma{}_{}.pdf'.format(sample,sigma,index))
        else:    
            plt.savefig('exp/filt/gauss/img_filt_gaussian_sigma{}_sigma{}_{}.pdf'.format(sample,sigma,index))
    
    return img_filtered


def median_filter(image, size, sample="n/a", show="False", index=0):
    """
    Description:    Applies a median filter to an imagee (array) and returns the filtered image 
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input image that need to be filtered 
    
    size:           float
                    Defines the width of the median filter, i.e. how many values are taken to build the 
                    median

    sample:         string
                    Defining which sample was choosen (default:"n/a")
    
    show:           bool
                    Defines whether or not the result is plotted (default = "False") 
    
    index:          int
                    Index of the iteration to label the images (default:0)    
    
    Outputs:
    --------
    img_filtered:   array
                    The filtered image
    """
    # Import packages needed for the function
    import matplotlib.pyplot as plt
    import scipy.ndimage as ndimage
    import os
    
    # Apply Gaussian filter to the gray image to reduce noise and show it to the user
    img_filtered = ndimage.median_filter(image, size)
    
    # Plot filtered image
    if show == "True":
        plt.imshow(img_filtered, cmap='gray', interpolation='nearest')
        plt.title('filtered image, Median (sample: {}, size: {}, index: {})'.format(sample,size,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')
        plt.show()
    
        # save it to a *.pdf file
        if not os.path.exists('exp/filt'):
            os.mkdir('exp/filt')
        if not os.path.exists('exp/filt/median'):
            os.mkdir('exp/filt/median')
            plt.savefig('exp/filt/median/img_filt_median_size{}_size{}_{}.pdf'.format(sample,size,index))
        else:    
            plt.savefig('exp/filt/median/img_filt_median_size{}_size{}_{}.pdf'.format(sample,size,index))
    
    return img_filtered


def crop_to_roi(image, sample="n/a", sigma=1.6, padding=0, show="False", index=0):
    """
    Description:    Reduces image to the region of interest (ROI), i.e. the section with  
    ------------    the polymer tube
    
    Inputs:
    -------
    image:          array
                    The input image that need to be processed 

    sample:         string
                    Defining which sample was choosen (default:"n/a")
                    
    sigma:          float
                    Defines the standard deviation of the edge detector (default: 1.6)
    
    padding:        int
                    Defines the padding widthof the image (default: 40)
    
    show:           bool
                    Defines whether or not the result is plotted (default: "False") 

    index:          int
                    Index of the iteration to label the images (default:0)
                    
    Outputs:
    --------
    reduced_img:    array
                    The reduced image with the specified padding
    """
    # Import packages needed for the function
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from skimage import data, feature, color
    from skimage.color import rgb2gray
    from skimage.feature import canny
    
    # Apply Canny edge detector to the gray image
    img_canny = feature.canny(color.rgb2gray(image), sigma)

    # compute offsets of fullsize image to region of interest 
    edge_indices = np.where(img_canny)
    position_y_min = min(edge_indices[0])
    position_y_max = max(edge_indices[0])
    position_x_min = min(edge_indices[1])
    position_x_max = max(edge_indices[1])

    # Calculating outer diameter based on canny edges
    d_outer_x = position_x_max - position_x_min
    d_outer_y = position_y_max - position_y_min
     
    # copy reduced image       
    reduced_img = image[position_y_min-padding:position_y_min+d_outer_y+padding, position_x_min-padding:position_x_min+d_outer_x+padding]

    # Create subplot to print reduced image and the marker for the outer diameter
    if show == "True":
        fig, ax = plt.subplots()
        ax.imshow(reduced_img, cmap='gray', interpolation='nearest')
        plt.title('Region of interest (sample: {}, index: {})'.format(sample,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')    
        plt.show()
        
        # save it to a *.pdf file
        if not os.path.exists('exp/roi'):
            os.mkdir('exp/roi')
            plt.savefig('exp/roi/img_roi{}_{}.pdf'.format(sample,index))
        else:    
            plt.savefig('exp/roi/img_roi{}_{}.pdf'.format(sample,index))
    
    return reduced_img


def find_diameter(image, sample="n/a", sigma=2.5, padding=40, show="False", index=0, border_col="red", border_width=2):
    """
    Description:    Finds diameter on the input image an plots the segmentation if needed
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input image that need to be processed

    sample:         string
                    Defining which sample was choosen (default:"n/a")
    
    sigma:          float
                    Defines the standard deviation of the edge detector (default: 5)
    
    padding:        int
                    Defines the padding width of the image (default: 40)
    
    show:           bool
                    Defines whether or not the result is plotted (default: "False")
    
    index:          int
                    Index of the iteration to label the images (default:0)    
    
    border_col:     string
                    Defines the color of the border (default: "red")
                    
    border_width:   int
                    Defines the width of the border (default: 2)                
    
    Outputs:
    --------
    d_outer:        tuple
                    Outer diameter of the tube in x and y direction
    
    center:         tuple
                    center point of the circle x and y coordinate                             
    """
    # Import packages needed for the function
    import numpy as np
    import matplotlib.pyplot as plt
    import os 
    from skimage import data, feature, color
    from skimage.color import rgb2gray
    from skimage.feature import canny
    from matplotlib.patches import Ellipse

    
    # Apply Canny edge detector to the gray image
    img_canny = feature.canny(color.rgb2gray(image), sigma)
    
    # compute offsets of fullsize image to region of interest 
    edge_indices = np.where(img_canny)
    position_y_min = min(edge_indices[0])
    position_y_max = max(edge_indices[0])
    position_x_min = min(edge_indices[1])
    position_x_max = max(edge_indices[1])

    # Calculating outer diameter based on canny edges
    d_outer_x = position_x_max - position_x_min
    d_outer_y = position_y_max - position_y_min
    
    # Create tuple to combine x and y values
    d_outer = [d_outer_x,d_outer_y]
    
    # Calculating the mean outer diameter
    d_outer_mean = np.mean(d_outer)

    #Calculate center point
    center = [(position_x_max - position_x_min)/2,(position_y_max - position_y_min)/2]
    
    # Create subplot to print reduced image and the marker for the inner diameter
    if show == "True":
        fig, ax = plt.subplots()
        ellipse = Ellipse((center[0]+padding,center[1]+padding),
                          width=d_outer_x,
                          height=d_outer_y,
                          edgecolor=border_col,
                          linewidth=border_width,
                          facecolor="none")
        ax.imshow(image, 
                  cmap='gray', 
                  interpolation='nearest')
        ax.add_artist(ellipse)
        plt.title('Diameter of the tube (sample: {}, d: {}, index: {})'.format(sample,d_outer_mean,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel') 
        plt.show()
        
        # save it to a *.pdf file
        if not os.path.exists('exp/diameter'):
            os.mkdir('exp/diameter')
            fig.savefig('exp/diameter/img_outerdiameter{}_{}.pdf'.format(sample,index))
        else:    
            fig.savefig('exp/diameter/img_outerdiameter{}_{}.pdf'.format(sample,index))
    
    #Return figure, tuples d_outer=[d_outer_x,d_outer_y], center=[x,y]
    return d_outer, center

def find_radii(image, sample="n/a", res=1, sigma=2.6, show="False", index=0):
    """
    Description:    Iterates around throught the found edges and calculates the distance to the center point, i.e. radius
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input image that need to be processed

    sample:         string
                    Defining which sample was choosen (default:"n/a")
    
    res:            float
                    Inter- and inner image resolution in in SI-unit (m) (default:"n/a")
                    
    sigma:          float
                    Defines the standard deviation of the edge detector (default: 2.6)
    
    show:           string
                    Defines whether or not the result is plotted (default: "False")
    
    index:          int
                    Index of the iteration to label the images (default:0)                   
    
    Outputs:
    --------
    radius:         list
                    list that contains all the radii of the image
    
    index:          int
                    The index of the image that was processed                             
    """
    # Import packages needed for the function
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from skimage import data, feature, color
    
    # Initialisation of the list to store the radius
    radius=[]
    
    # Convert given image into array
    image = color.rgb2gray(np.asarray(image))

    # Edge detection wih canny edge detector
    img_edges = feature.canny(image, sigma=sigma)

    # Create an array with the coordinates of the edges
    edge_indices = np.where(img_edges)

    # Calculate the center position
    center_x = (max(edge_indices[0])-min(edge_indices[0]))/2
    center_y = (max(edge_indices[1])-min(edge_indices[1]))/2

    # Compose tuple to store center point in x and y direction
    center_coordinates = [center_x,center_y]

    ## Loop to calculate the radius of the edge.
    for w in range(len(edge_indices[0])):
        for h in range(len(edge_indices[1])):
            circle_coordinates = [edge_indices[0][w],edge_indices[1][h]]
            #print(circle_coordinates)
            r_vec = [circle_coordinates[0]-center_coordinates[0],circle_coordinates[1]-center_coordinates[1]]
            #print(r_vec)
            r_abs = np.sqrt(np.square(r_vec[0])+np.square(r_vec[1]))
            radius.append(r_abs)
            #print("Radius (absolut, length): ",r_abs)
    
    # Create plot to print output image if show == "True"
    if show == "True":
        plt.imshow(img_edges, cmap='gray', 
                   interpolation='nearest')
        plt.title('Edges found with Canny-Edge detector (sigma: {}, index: {})'.format(sigma,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')
        plt.show() 
        
    #Return the list with all the radii in the given image and the index of the given image
    return radius, index

    
def calculate_wallthickness (d_outer, center_outer, d_inner, center_inner):
    """
    Description:    Calculates wallthickness based on the outer and inner diameter and center point informations
    ------------    
    
    Inputs:
    -------
    d_outer:        tuple
                    Contains the outer diameter in x and y direction 
    
    center_outer:   tuple
                    Contains the center point coordinates (x and y) of the outer diameter
    
    d_inner:        tuple
                    Contains the inner diameter in x and y direction
    
    center_inner:   tuple
                    Contains the center point coordinates (x and y) of the inner diameter
    
    Outputs:
    --------
    wallthicknesses:vector
                    The numerical values of the wall thickness in x and y direction and the mean values
    
    """
    # Import packages needed for the function
    import numpy as np
    
    # Wallthickness in x-direction at 0° and 180°
    wallthickness_x_0 = (center_inner[0]-(center_inner[0]+d_inner[0]/2))-(center_outer[0]-(center_outer[0]+d_outer[0]/2))  
    wallthickness_x_180 = (center_outer[0]-(center_outer[0]-d_outer[0]/2))-(center_inner[0]-(center_inner[0]-d_inner[0]/2))
    
    # Wallthickness in y-direction at 90° and 270°
    wallthickness_y_90 = (center_inner[1]-(center_inner[1]+d_inner[1]/2))-(center_outer[1]-(center_outer[1]+d_outer[1]/2))  
    wallthickness_y_270 = (center_outer[1]-(center_outer[1]-d_outer[1]/2))-(center_inner[1]-(center_inner[1]-d_inner[1]/2))
    
    # Calculating mean wall thickness in x-and y-direction
    wallthickness_x_mean = (d_outer[0]-d_inner[0])/2
    wallthickness_y_mean = (d_outer[1]-d_inner[1])/2
        
    #return wall thicknesses as tuple
    return [wallthickness_x_0, wallthickness_x_180, wallthickness_y_90, wallthickness_y_270, wallthickness_x_mean, wallthickness_y_mean]
    

def plot_segmentation(image, d_outer, center_outer, d_inner, center_inner, padding=40, sample="n/a", show="False", index=0, border_col="red", border_width=2):
    """
    Description:    Plots an image with two circles on it, based on outer and inner diameter and the two center points.
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input image that need to be filtered
    
    d_outer:        tuple
                    Contains the outer diameter in x and y direction 
    
    center_outer:   tuple
                    Contains the center point coordinates (x and y) of the outer diameter
    
    d_inner:        tuple
                    Contains the inner diameter in x and y direction
    
    center_inner:   tuple
                    Contains the center point coordinates (x and y) of the inner diameter

    sample:         string
                    Defining which sample was choosen (default:"n/a")
                    
    index:          int
                    Index of the iteration to label the images (default:0)                    
    
    Outputs:
    --------
    wallthickness:  tuple
                    The numerical values of the wall thickness in x and y direction
    
    """
    # Import packages needed for the function
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import os
    
    # Calculate the outer diameter in x and y direction and mean outer diameter
    d_outer_x = d_outer[0]
    d_outer_y = d_outer[1]
    d_outer_mean = np.mean([d_outer[0],d_outer[1]])
    
    # Calculate the inner diameter in x and y direction and mean inner diameter
    d_inner_x = d_inner[0]
    d_inner_y = d_inner[1]
    d_inner_mean = np.mean([d_inner[0],d_inner[1]])
    
    # Calculating wallthicknesses and mean wallthickness of the tube
    t_x = (d_outer[0]-d_inner[0])/2
    t_y = (d_outer[1]-d_inner[1])/2
    t_mean = np.mean([t_x,t_y])
    
    # Create subplot to print reduced image and the marker for the outer and inner diameter
    if show == "True":
        fig, ax = plt.subplots()
        ellipse_inner = Ellipse((center_inner[0]+padding+t_x,center_inner[1]+padding+t_y),
                                width=d_inner_x,
                                height=d_inner_y,
                                edgecolor=border_col,
                                linewidth=border_width,
                                facecolor="none")
        ellipse_outer = Ellipse((center_outer[0]+padding,center_outer[1]+padding),
                                width=d_outer_x,
                                height=d_outer_y,
                                edgecolor=border_col,
                                linewidth=border_width,
                                facecolor="none")
        ax.imshow(image, cmap='gray', interpolation='nearest')
        ax.add_artist(ellipse_outer)
        ax.add_artist(ellipse_inner)
        plt.title('Segement of the tube (sample: {}, d[i]:{}, d[o]:{}, index: {})'.format(sample, d_inner_mean,d_outer_mean,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')    
        plt.show()
    
        # save it to a *.pdf file
        if not os.path.exists('exp/segementation'):
            os.mkdir('exp/segementation')
            fig.savefig('exp/segementation/img_full_segmentation{}_{}.pdf'.format(sample,index))
        else:    
            fig.savefig('exp/segementation/img_full_segmentation{}_{}.pdf'.format(sample,index))
    
    return d_outer_mean, d_inner_mean


def invert_image(image, sample="n/a", show="False", index=0):
    """
    Description:    Convert the given image to a gray scale image and invert it
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input imaget that need to be filtered                

    sample:         string
                    Defining which sample was choosen (default:"n/a")
                    
    show:           bool
                    Defines whether or not the result is plotted (default: "False") 

    index:          int
                    Index of the iteration to label the images (default:0)
                    
    Outputs:
    --------
    inverted_image: array
                    Interverted image
                             
    """
    # Import packages needed for the function
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from skimage import data, feature, color
    from skimage.color import rgb2gray
    
    # Invert image with np.invert()
    inverted_image = np.invert(color.rgb2gray(image))
    
    # Create subplot to print reduced image and the marker for the outer diameter
    if show == "True":
        plt.imshow(inverted_image, cmap='gray', 
                   interpolation='nearest')
        plt.title('Inverted image (sample: {}, index: {})'.format(sample,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')
        plt.show()
        
        # save it to a *.pdf file
        if not os.path.exists('exp/invert'):
            os.mkdir('exp/invert')
            plt.savefig('exp/invert/img_inverted{}_{}.pdf'.format(sample,index))
        else:    
            plt.savefig('exp/invert/img_inverted{}_{}.pdf'.format(sample,index))
        
    #Return inverted image
    return inverted_image


def mask_image(image, sides, center, mode="circle", padding=40, show="False", sample="n/a", index=0):
    """
    Description:    Masks image based on d_outer and center_outer & outputs the new image
    ------------    
    
    Inputs:
    -------
    image:          array
                    The input imaget that need to be masked                
    
    sides:          Tuple
                    Contains the outer dimensions in x- and y direction for masking 
    
    center:         tuple
                    Contains the center point coordinates (x and y) of the outer dimension
    
    mode:           string
                    Defining the mode for the masking "circle", "ellipse" or "square" (default: "circle")
    
    sample:         string
                    Defining which sample was choosen (default:"n/a")
                    
    show:           bool
                    Defines whether or not the result is plotted (default: "False") 

    index:          int
                    Index of the iteration to label the images (default:0)
                    
    Outputs:
    --------
    masked image:   array
                    Interverted image
                             
    """
    # Import packages needed for the function
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import data, feature, color
    from skimage.color import rgb2gray

    # Create copy of the input image and convert it to gray-scale for image manipulation
    image = image.copy()
    image = color.rgb2gray(image)
    
    # Extract the side dimensions from the tuple sides
    side_x = sides[0]
    side_y = sides[1]
    
    #Calculate the minimum diameter
    side = np.minimum(sides[0],sides[1])
    
    # Mode "circle"
    if mode == "circle":
        ## Nested for-loop to reduce image information to the inner tube lumen (circle)
        for h in range(image.shape[1]):
            for w in range(image.shape[0]):
                ### Circle formula outside the outer diameter (side), replace values with 1 (=white) 
                if (np.sqrt(np.square(w-center[0]-padding) + np.square(h-center[1]-padding)) >= (side-25)/2):
                    image[w, h] = 1
            
                ### Circle formula inside the outer diameter (side), keep the image values 
                elif (np.sqrt(np.square(w-center[0]-padding) + np.square(h-center[1]-padding)) < (side-25)/2):
                    image[w, h] = image[w, h]
    
        ## Assign new image to a new variable
        masked_image = image
    
    # Mode "ellipse"
    if mode == "ellipse":
        ## Nested for-loop to reduce image information to the inner tube lumen (ellipse)
        for h in range(image.shape[1]):
            for w in range(image.shape[0]):
                ### Ellipse formula outside the side dimenstions, replace values with 1 (=white) 
                if (np.square(h-center[0]-padding)/np.square((side_x-25)/2) + np.square(w-center[1]-padding)/np.square((side_y-25)/2) >= 1):
                    image[w, h] = 1
            
                ### Ellipse formula inside the side dimensions, keep the image values 
                elif (np.square(h-center[0]-padding)/np.square((side_x-25)/2) + np.square(w-center[1]-padding)/np.square((side_y-25)/2) < 1):
                    image[w, h] = image[w, h]
        
        ## Assign new image to a new variable
        masked_image = image
    
    # Mode "square"
    if mode == "square":
        ## Nested for-loop to reduce image information to the inner tube lumen (squared)
        for h in range(image.shape[1]):
            for w in range(image.shape[0]):
                ### Remove information around the given side width and hight, replace values with 1 (=white) 
                if (w >= (center[0]+padding)+(side-25)/2 or h >= (center[0]+padding)+(side-25)/2):
                    image[w, h] = 1
             
                elif (w <= (center[0]+padding)-(side-25)/2 or h <= (center[0]+padding)-(side-25)/2):
                    image[w, h] = 1
    
        ## Assign new image to a new variable
        masked_image = image
    
    # Create plot to print output image if show == "True"
    if show == "True":
        plt.imshow(image, cmap='gray', 
                   interpolation='nearest')
        plt.title('Reduced image to inner lumen (sample: {}, index: {})'.format(sample,index))
        plt.xlabel('Image width / pixel')
        plt.ylabel('Image hight / pixel')
        plt.show()         
    
    # Return the maske image to the function call
    return masked_image 


def export_results(z, d_o_x, d_o_y, d_o, r_o, d_i_x, d_i_y, d_i, r_i, t_x_0, t_x_180, t_y_90, t_y_270 ,t_mean, res="n/a", sample="n/a", index=0, file="n/a"):
    """
    Description:    Exports the results into an .txt file. A result summary is printed to the console promt
    ------------    
    
    Inputs:
    -------
    z:              float
                    Tube length in SI-unit (m)                
    
    d_o_x, d_o_y:   float
                    Outer diameter of the tube in x and y direction in SI-unit (m)  
    
    d_o:            float
                    Mean outer diameter of the tube in SI-unit (m)
    
    r_o:            float
                    Mean outer radius of the tube in SI-unit (m)
    
    d_i_x, d_i_y:   float
                    Inner diameter of the tube in x and y direction in SI-unit (m) 
                    
    d_i:            float
                    Mean inner diameter of the tube in in SI-unit (m) 
 
    r_i:            float
                    Mean inner radius of the tube in SI-unit (m)
                    
    t_x_0:          float
                    Wall thickness at an angle of 0° in SI-unit (m)
    
    t_x_180:        float
                    Wall thickness at an angle of 180° in SI-unit (m)
    
    t_y_90:         float
                    Wall thickness at an angle of 90° in SI-unit (m)
    
    t_y_270:        float
                    Wall thickness at an angle of 270° in SI-unit (m)

    t_mean:         float
                    Wall thickness in SI-unit (m)
           
    time_c:         float
                    Computation time in seconds
                    
    res:            float
                    Inter- and inner image resolution in in SI-unit (m) (default:"n/a")
    
    sample:         string
                    Index of the iteration to label the images (default:"n/a")
    
    Outputs:
    --------
    none                              
    """
    # Import packages needed for the function
    import os

    # save results to a *.txt file
    if not os.path.exists('res/txt'):
        os.mkdir('res/txt')
        
    if(index==0):
            ## Print Header
            file.write("idx," + "sample," + "l," + 
                       "d_o_x," + "d_o_y," + "d_o," + "r_o," + 
                       "d_i_x," + "d_i_y," + "d_i," + "r_i," + 
                       "t_x_0," + "t_x_180," + "t_y_90," + "t_y_270," +
                       "t_mean\n")
            file.write(str(index) + "," +
                       str(sample) + "," +
                       str(round(z,2)) + "," + 
                       str(round(d_o_x,2))  + "," + 
                       str(round(d_o_y,2))  + "," + 
                       str(round(d_o,2))    + "," + 
                       str(round(r_o,2))    + "," + 
                       str(round(d_i_x,2))  + "," + 
                       str(round(d_i_y,2))  + "," + 
                       str(round(d_i,2))    + "," + 
                       str(round(r_i,2))    + "," +
                       str(round(t_x_0,2))  + "," +
                       str(round(t_x_180,2))+ "," +
                       str(round(t_y_90,2)) + "," +
                       str(round(t_y_270,2))+ "," +
                       str(round(t_mean,2)) + "\n")
                
    else:
        file.write(str(index) + "," +
                   str(sample) + "," +
                   str(round(z,2)) + "," + 
                   str(round(d_o_x,2))  + "," + 
                   str(round(d_o_y,2))  + "," + 
                   str(round(d_o,2))    + "," + 
                   str(round(r_o,2))    + "," + 
                   str(round(d_i_x,2))  + "," + 
                   str(round(d_i_y,2))  + "," + 
                   str(round(d_i,2))    + "," + 
                   str(round(r_i,2))    + "," +
                   str(round(t_x_0,2))  + "," +
                   str(round(t_x_180,2))+ "," +
                   str(round(t_y_90,2)) + "," +
                   str(round(t_y_270,2))+ "," +
                   str(round(t_mean,2)) + "\n")
    
    
def print_summary(z, d_o_x, d_o_y, d_o, r_o, d_i_x, d_i_y, d_i, r_i, t, time_c, sample="n/a", res="n/a", index=0):
    """
    Description:    Print summary to the console with key indicators of the calculation process
    ------------    
    
    Inputs:
    -------
    z:              float
                    Tube length in SI-unit (m)                
    
    d_o_x, d_o_y:   float
                    Outer diameter of the tube in x and y direction in SI-unit (m)  
    
    d_o:            float
                    Mean outer diameter of the tube in SI-unit (m)
    
    r_o:            float
                    Mean outer radius of the tube in SI-unit (m)
    
    d_i_x, d_i_y:   float
                    Inner diameter of the tube in x and y direction in SI-unit (m) 
                    
    d_i:            float
                    Mean inner diameter of the tube in in SI-unit (m) 
 
    r_i:            float
                    Mean inner radius of the tube in SI-unit (m)
                    
    t:              float
                    Wall thickness in SI-unit (m)
           
    time_c:         float
                    Computation time in seconds
                    
    res:            float
                    Inter- and inner image resolution in in SI-unit (m) (default:"n/a")
    
    sample:         string
                    Index of the iteration to label the images (default:"n/a")
    
    Outputs:
    --------
    nothing is returned, but a summary is printed into console                              
    """
    # Calculate minimum, maximum and difference values
    t_min = min(t)
    t_max = max(t)
    t_diff = t_max-t_min
    d_o_min = min(d_o)
    d_o_max = max(d_o)
    d_o_x_max = max(d_o_x)
    d_o_y_max = max(d_o_y)
    d_i_min = min(d_i)
    d_i_max = max(d_i)
    d_i_x_max = max(d_i_x)
    d_i_y_max = max(d_i_y)

    # Print values in promt
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("% RESULT SUMMARY (" +str(sample) + "): \n% --------------------------")
    print("% # Inner image resolution: " + str(res) + " um")
    print("% # Inter image resolution: " + str(res) + " um")
    print("% # Total computation time: " + str(time_c) + " sec")
    print("% ")
    print("% Outer diameter:")
    print("% - Maximum outer diameter in x-direction: {} um".format(round(d_o_x_max, 2)))
    print("% - Maximum outer diameter in y-direction: {} um".format(round(d_o_y_max, 2)))
    print("% - Minimum mean outer diameter: {} um".format(round(d_o_min, 2)))
    print("% - Maximum mean outer diameter: {} um".format(round(d_o_max, 2)))
    print("% ")
    print("% Inner diameter:")
    print("% - Maximum inner diameter in x-direction: {} um".format(round(d_i_x_max, 2)))
    print("% - Maximum inner diameter in y-direction: {} um".format(round(d_i_y_max, 2)))
    print("% - Minimum mean inner diameter: {} um".format(round(d_i_min, 2)))
    print("% - Maximum mean inner diameter: {} um".format(round(d_i_max, 2)))
    print("% ")
    print("% Wall thickness:")
    print("% - Minimum mean wall thickness: {} um".format(round(t_min, 2)))
    print("% - Maximum mean wall thickness: {} um".format(round(t_max, 2)))
    print("% - Variation (max-min): {} um".format(round(t_diff, 2)))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")