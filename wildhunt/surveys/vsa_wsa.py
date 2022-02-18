import time
import os
import numpy as np
from astropy.io import fits
import multiprocessing
from multiprocessing import Process, Queue
from IPython import embed

# Use single cores (forcing it for numpy operations)
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Survey parameters
survey_param = {'archive':'WSA','database':'UKIDSSDR11PLUS','programmeID':'101','bands':('Y','H','K'),'idPresent':'noID',
                'userX':'0.5', 'email':'','email1':'','crossHair':'n','mode':'wget'}

numCoords=10000  # number of coords in each bach (do not exceed 20000)

boundary="--FILEUPLOAD"  # separator

def deg2hms(ra, dec):
    rah = np.floor(ra / 15.0)
    ram = np.floor((ra / 15.0 - rah) * 60.0)
    ras = ((ra / 15.0 - rah) * 60.0 - ram) * 60.0
    if dec < 0:
        dec = np.abs(dec)
        decd = 0 - np.floor(dec)
        decm = np.floor((dec + decd) * 60.0)
        decs = ((dec + decd) * 60.0 - decm) * 60.0
    else:
        decd = np.floor(dec)
        decm = np.floor((dec - decd) * 60.0)
        decs = ((dec - decd) * 60.0 - decm) * 60.0
    ra = [rah, ram, ras]
    dec = [decd, decm, decs]
    return ra, dec

def hms2name(ra, dec):
    rahms, decdms = deg2hms(ra, dec)
    rahd = int(rahms[0])
    ramd = int(rahms[1])
    rasd = round(rahms[2], 3)

    rastr = ('{rahs:02d}''{rams:02d}''{rass:6.3f}')
    ras = rastr.format(rahs=rahd, rams=ramd, rass=rasd)
    rass = ras.replace(' ', '0')

    dechd = int(np.abs(decdms[0]))
    decmd = int(decdms[1])
    decsd = round(decdms[2], 3)

    decstr = ('{dechs:02d}''{decms:02d}''{decss:6.3f}')
    decs = decstr.format(dechs=dechd, decms=decmd, decss=decsd)
    decss = decs.replace(' ', '0')
    if dec < 0:
        decss = '-' + decss
    else:
        decss = '+' + decss

    name = 'J' + rass + decss
    namebrief = name[0:5] + name[11:-6]
    return name, namebrief

def download_links(ra,dec):
    '''
    Download the links to the images and produce all the *_wget.sh files
    '''
    if np.size(ra)>numCoords:
        nbatch=int(np.ceil(np.size(ra)/numCoords))
    else:
        nbatch=1

    loop = 0
    for i in range(nbatch):
        startID=numCoords * loop
        loop+=1
        uploadFile="upload_"+str(loop)+".txt"
        outFile="out_"+str(loop)+".txt"
        up = open(uploadFile, 'w+')

        for j in survey_param:
            if j=='bands':
                for band in (survey_param[j]):
                    print(boundary, file=up)
                    print('Content-Disposition: form-data; name=\"band\"\n', file=up)
                    print(band+' ', file=up)
            else:
                print(boundary,file=up)
                print('Content-Disposition: form-data; name=\"{}\"\n'.format(j),file=up)
                print(str(survey_param[j])+' ',file=up)
        print(boundary, file=up)
        print('Content-Disposition: form-data; name=\"startID\"\n', file=up)
        print(str(startID)+' ', file=up)
        print(boundary, file=up)
        print('Content-Disposition: form-data; name=\"fileName\"; filename=\"{}\" Content-Type: text/plain\n'.format(uploadFile), file=up)

        if i!=(nbatch-1):
            n=numCoords
        else:
            n=len(ra)-startID
        for j in range(n):
            if dec[startID+j]>=0:
                print('{:>15},{:>15}'.format('+'+str(ra[startID + j]), '+'+str(dec[startID + j])), file=up)
            else:
                print('{:>15},{:>15}'.format('+'+str(ra[startID+j]), str(dec[startID+j])), file=up)
        print('\n',file=up)
        print(boundary+'--', file=up)
        up.close()

        os.system("wget --keep-session-cookies --header=\"Content-Type: multipart/form-data;  boundary=FILEUPLOAD\""
                  " --post-file {} http://wsa.roe.ac.uk:8080/wsa/tmpMultiGetImage -O {}".format(uploadFile,outFile))

        out = open(outFile,'r')
        Lines = out.readlines()

        dlFile = str(loop) + '_wget.sh'
        sh = open(dlFile, 'w')

        j=0
        n_bands=1
        for line in Lines:
            record='--http'
            if (record in line):
                name, namebrief=hms2name(ra[startID + j],dec[startID + j]) # Name of the images to download
                if n_bands==len(survey_param['bands']):
                    j+=1
                    n_bands=0
                n_bands+=1
                strtPos=line.index(record)
                endPos=line.rindex('-->')
                fileURL=line[strtPos+2:endPos]
                if ('band=NULL' in line):
                    print('Skip download')
                else:
                    band_url=line.index('band=')+len('band=')
                    name_url=name+'.'+line[band_url]+'.fits.gz'
                    sh.write("wget \"{}\" -O {}\n".format(fileURL,name_url))
        out.close()
        sh.close()
        print("Start sleep")
        time.sleep(5)
        print("End sleep")

def download_parallel(ra,n_process=16):
    '''
        Perform forced photometry for several sources
        Args:
            coadd_id:
            ra_id:
            dec_id:
            n_process
    '''
    if np.size(ra)>numCoords:
        nbatch=int(np.ceil(np.size(ra)/numCoords))
    else:
        nbatch=1
    n_file = nbatch
    n_cpu = multiprocessing.cpu_count()

    if n_process>n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file

    work_queue = Queue()
    processes = []

    for ii in range(n_file):
        work_queue.put([ii])

    for w in range(n_process):
        p = Process(target=download_images, args=(work_queue,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def download_images(work_queue):
    '''
    Execute the *_wget.sh files to download the images
    '''
    while not work_queue.empty():
        nbatch = work_queue.get()[0]
        os.system("bash {}_wget.sh".format(str(nbatch+1)))
        sh=open('{}_finished.txt'.format(str(nbatch+1)),'w')
        sh.close()

def gzip_parallel(ra,dec,n_process=16):
    '''
        Unzip images for several sources in parallel
        Args:
            coadd_id:
            ra_id:
            dec_id:
            n_process
    '''
    n_file = np.size(ra)
    n_cpu = multiprocessing.cpu_count()

    if n_process>n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file

    work_queue = Queue()
    processes = []

    for ii in range(n_file):
        work_queue.put([ra[ii],dec[ii]])

    for w in range(n_process):
        p = Process(target=unzip_images, args=(work_queue,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def unzip_images(work_queue):
    while not work_queue.empty():
        ra, dec = work_queue.get()
        name, namebrief = hms2name(ra, dec)
        print("Unzipping {} source".format(name))
        os.system("gzip -d -f {}.Y.fits.gz".format(name))
        os.system("gzip -d -f {}.H.fits.gz".format(name))
        os.system("gzip -d -f {}.K.fits.gz".format(name))

# Main code to launch the software
start_time = time.time()
par = fits.open('total_noDELSDR9_2arcsec_Jselected_UKIDSS_DELS_filtered.fits')
#par = fits.open('UKIDSS_subsample.fits')
data = par[1].data
ra, dec = data['ra'], data['dec']
nbatch=download_links(ra,dec)
download_parallel(ra,n_process=5)
gzip_parallel(ra,dec,n_process=25)

print("Elapsed time: {0:.2f} sec" .format(time.time() - start_time))
f = open('time_spent.txt','w')
print(time.time()-start_time,file=f)

