import time
import os
import numpy as np
import multiprocessing
from multiprocessing import Process, Queue
from wildhunt.surveys import imagingsurvey
from IPython import embed

# List of programmeID: VHS - 110, VVV - 120, VMC - 130, VIKING - 140, VIDEO - 150, UHS - 107, UKIDSS: LAS - 101,
# GPS - 102, GCS - 103, DXS - 104, UDS - 105
numCoords=10#000  # number of coords in each bach (do not exceed 20000)

class VsaWsa(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, name, verbosity=1):
        """
        :param bands:
        :param fov:
        :param name:
        """
        if name[0] == 'U':
            archive='WSA'
            if name[0:3] == 'UKI':
                if name[-3::] == 'LAS': programID='101'
                if name[-3::] == 'GPS': programID = '102'
                if name[-3::] == 'GCS': programID = '103'
                if name[-3::] == 'DXS': programID = '104'
                if name[-3::] == 'UDS': programID = '105'
                name=name[:-3]
            if name[0:3] == 'UHS': programID = '107'
        else:
            archive='VSA'
            if name[0:3]=='VHS': programID='110'
            if name[0:3] == 'VVV': programID = '120'
            if name[0:3] == 'VMC': programID = '130'
            if name[0:3] == 'VIK': programID = '140'
            if name[0:3] == 'VID': programID = '150'

        self.archive=archive
        self.programID=programID

        super(VsaWsa, self).__init__(bands, fov, name, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs):
        """
        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """
        # Check if download directory exists. If not, create it
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)

        self.download_links(ra,dec,image_folder_path)
        self.download_parallel(ra, n_jobs)
        os.system('rm *_wget.sh')
        os.system('rm out_*')
        os.system('rm upload_*')
        os.system('rm finished_*')

    def download_links(self, ra, dec, path):
        '''
        Download the links to the images and produce all the *_wget.sh files
        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        '''

        # Survey parameters
        survey_param = {'archive': self.archive, 'database': self.name, 'programmeID': self.programID,
                        'bands': self.bands,
                        'idPresent': 'noID', 'userX': '0.5', 'email': '', 'email1': '', 'crossHair': 'n',
                        'mode': 'wget'}
        boundary = "--FILEUPLOAD"  # separator

        if np.size(ra) > numCoords:
            nbatch = int(np.ceil(np.size(ra) / numCoords))
        else:
            nbatch = 1

        loop = 0
        for i in range(nbatch):
            startID = numCoords * loop
            loop += 1
            uploadFile = "upload_" + str(loop) + ".txt"
            outFile = "out_" + str(loop) + ".txt"
            up = open(uploadFile, 'w+')

            for j in survey_param:
                if j == 'bands':
                    for band in (survey_param[j]):
                        print(boundary, file=up)
                        print('Content-Disposition: form-data; name=\"band\"\n', file=up)
                        print(band + ' ', file=up)
                else:
                    print(boundary, file=up)
                    print('Content-Disposition: form-data; name=\"{}\"\n'.format(j), file=up)
                    print(str(survey_param[j]) + ' ', file=up)
            print(boundary, file=up)
            print('Content-Disposition: form-data; name=\"startID\"\n', file=up)
            print(str(startID) + ' ', file=up)
            print(boundary, file=up)
            print(
                'Content-Disposition: form-data; name=\"fileName\"; filename=\"{}\" Content-Type: text/plain\n'.format(
                    uploadFile), file=up)

            if i != (nbatch - 1):
                n = numCoords
            else:
                n = len(ra) - startID
            for j in range(n):
                if dec[startID + j] >= 0:
                    print('{:>15},{:>15}'.format('+' + str(ra[startID + j]), '+' + str(dec[startID + j])), file=up)
                else:
                    print('{:>15},{:>15}'.format('+' + str(ra[startID + j]), str(dec[startID + j])), file=up)
            print('\n', file=up)
            print(boundary + '--', file=up)
            up.close()

            if self.archive=='WSA':
                os.system(
                    "wget --keep-session-cookies --header=\"Content-Type: multipart/form-data;  boundary=FILEUPLOAD\""
                      " --post-file {} http://wsa.roe.ac.uk:8080/wsa/tmpMultiGetImage -O {}".format(uploadFile,
                                                                                                    outFile))
            else:
                os.system(
                    "wget --keep-session-cookies --header=\"Content-Type: multipart/form-data;  boundary=FILEUPLOAD\""
                    " --post-file {} http://horus.roe.ac.uk:8080/vdfs/MultiGetImage -O {}".format(uploadFile,
                                                                                                  outFile))

            out = open(outFile, 'r')
            Lines = out.readlines()

            dlFile = str(loop) + '_wget.sh'
            sh = open(dlFile, 'w')

            j = 0
            n_bands = 1
            for line in Lines:
                record = '--http'
                if (record in line):
                    name, namebrief = hms2name(ra[startID + j], dec[startID + j])  # Name of the images to download
                    if n_bands == len(survey_param['bands']):
                        j += 1
                        n_bands = 0
                    n_bands += 1
                    strtPos = line.index(record)
                    endPos = line.rindex('-->')
                    fileURL = line[strtPos + 2:endPos]
                    if ('band=NULL' in line):
                        print('Skip download')
                    else:
                        band_url = line.index('band=') + len('band=')
                        name_url = name + '.' + line[band_url] + '.fits'
                        sh.write("wget \"{}\" -O {}/{}\n".format(fileURL, path, name_url))
            out.close()
            sh.close()
            print("Start sleep")
            time.sleep(5)
            print("End sleep")

    def download_parallel(self, ra, n_process=1):
        '''
            Download many sources in parallel
            Args:
                coadd_id:
                ra_id:
                dec_id:
                n_process
        '''
        if np.size(ra) > numCoords:
            nbatch = int(np.ceil(np.size(ra) / numCoords))
        else:
            nbatch = 1
        n_file = nbatch
        n_cpu = multiprocessing.cpu_count()

        if n_process > n_cpu:
            n_process = n_cpu

        if n_process > n_file:
            n_process = n_file

        work_queue = Queue()
        processes = []

        for ii in range(n_file):
            work_queue.put([ii])

        for w in range(n_process):
            p = Process(target=self.download, args=(work_queue,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    def download(self, work_queue):
        '''
        Execute the *_wget.sh files to download the images
        '''
        while not work_queue.empty():
            nbatch = work_queue.get()[0]
            os.system("bash {}_wget.sh".format(str(nbatch + 1)))
            sh = open('{}_finished.txt'.format(str(nbatch + 1)), 'w')
            sh.close()

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

