"""
Propagation happening on the setup

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, Matthew O'Toole, G. Wetzstein.
SIGGRAPH 2022
"""

import torch
import torch.nn as nn
import utils
import time
import cv2
import imageio

from hw.phase_encodings import phase_encoding
import sys
if sys.platform == 'win32':
    import slmpy
    import hw.camera_capture_module as cam
    import hw.calibration_module as calibration


class PhysicalProp(nn.Module):
    """ A module for physical propagation,
    forward pass displays gets SLM pattern as an input and display the pattern on the physical setup,
    and capture the diffraction image at the target plane,
    and then return warped image using pre-calibrated homography from instantiation.

    Class initialization parameters
    -------------------------------
    :param params_slm: a set of parameters for the SLM.
    :param params_camera: a set of parameters for the camera sensor.
    :param params_calib: a set of parameters for homography calibration.
    :param q_fn: quantization function module

    Usage
    -----
    Functions as a pytorch module:

    >>> camera_prop = PhysicalProp(...)
    >>> captured_amp = camera_prop(slm_phase)

    slm_phase: phase at the SLM plane, with dimensions [batch, 1, height, width]
    captured_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]

    """
    def __init__(self, params_slm, params_camera, params_calib=None, q_fn=None, shutter_speed=100, hdr=False):
        super(PhysicalProp, self).__init__()


        self.shutter_speed = shutter_speed
        self.hdr = hdr
        self.q_fn = q_fn
        self.params_calib = params_calib

        if self.hdr:
            assert len(self.shutter_speed) > 1 # more than 1 shutter speed for HDR capture
        else:
            assert len(self.shutter_speed) == 1 # non-hdr mode supports only one shutter speed

        # 1. Connect Camera
        self.camera = cam.CameraCapture(params_camera)
        self.camera.connect(0)  # specify the camera to use, 0 for main cam, 1 for the second cam
        #self.camera.start_capture()
        self.camera.start_capture()

        # 2. Connect SLM
        self.slm = slmpy.SLMdisplay(isImageLock=True, monitor=params_slm.monitor_num)
        self.params_slm = params_slm

        # 3. Calibrate hardware using homography
        if params_calib is not None:
            self.warper = calibration.Warper(params_calib)
            self.calibrate(params_calib.phase_path, params_calib.show_preview)
        else:
            self.warper = None

    def calibrate_total_laser_energy(self):
        print("Calibrating total laser energy...")
        phase_img = imageio.imread(self.params_calib.blank_phase_path)
        self.slm.updateArray(phase_img)
        time.sleep(5)
        captured_plane_wave = self.forward(phase_img)
        h, w = captured_plane_wave.shape[-2], captured_plane_wave.shape[-1] # full SLM size
        cropped_energy = utils.crop_image(captured_plane_wave**2, (500, 500), stacked_complex=False)
        self.total_laser_energy = cropped_energy.sum() * (h * w) / (500 * 500)

    def calibrate(self, phase_path, show_preview=False):
        """

        :param phase_path:
        :param show_preview:
        :return:
        """
        print('  -- Calibrating ...')
        self.camera.set_shutter_speed(2000) # for homography pattern. remember to reset it!
        self.camera.set_gain(10) # for homography pattern. remember to reset it!
        phase_img = imageio.imread(phase_path)
        #print(phase_img)
        self.slm.updateArray(phase_img)
        time.sleep(5)
        captured_img = self.camera.grab_images_fast(5)  # capture 5-10 images for averaging
        calib_success = self.warper.calibrate(captured_img, show_preview)
        self.camera.set_gain(0)
        if calib_success:
            print('  -- Calibration succeeded!...')
            if not self.hdr:
                print("One time step shutter speed for non-HDR capture...")
                self.camera.set_shutter_speed(self.shutter_speed[0]) # reset for capture
        else:
            raise ValueError('  -- Calibration failed')

    def forward(self, slm_phase, time_avg=1):
        """

        :param slm_phase:
        :return:
        """
        input_phase = slm_phase
        if self.q_fn is not None:
            dp_phase = self.q_fn(input_phase)
        else:
            dp_phase = input_phase

        self.display_slm_phase(dp_phase)

        raw_intensity_sum = 0
        for t in range(time_avg):
            raw_intensity = self.capture_linear_intensity(dp_phase)  # grayscale raw16 intensity image
            raw_intensity_sum += raw_intensity
        raw_intensity = raw_intensity_sum / time_avg

        # amplitude is returned! not intensity!
        warped_intensity = self.warper(raw_intensity)  # apply homography
        return warped_intensity.sqrt()  # return amplitude

    def capture_linear_intensity(self, slm_phase):
        """
        display a phase pattern on the SLM and capture a generated holographic image with the sensor.

        :param slm_phase:
        :return:
        """
        raw_uint16_data = self.capture_uint16()  # display & retrieve buffer
        captured_intensity = self.process_raw_data(raw_uint16_data)  # demosaick & sum up
        return captured_intensity
    
    def forward_hdr(self, slm_phase):
        """

        :param slm_phase:
        :return:
        """
        input_phase = slm_phase
        if self.q_fn is not None:
            dp_phase = self.q_fn(input_phase)
        else:
            dp_phase = input_phase

        raw_intensity_hdr, raw_intensity_stack  = self.capture_linear_intensity_hdr(dp_phase)  # grayscale raw16 intensity image

        # amplitude is returned! not intensity!
        warped_intensity_hdr = self.warper(raw_intensity_hdr)  # apply homography
        warped_intensity_stack = [self.warper(intensity) for intensity in raw_intensity_stack]
        warped_amplitude_hdr = warped_intensity_hdr.sqrt()
        warped_amplitude_stack = [intensity.sqrt() for intensity in warped_intensity_stack]
        return warped_amplitude_hdr, warped_amplitude_stack

    def capture_linear_intensity_hdr(self, slm_phase):
        raw_uint16_data_list = []
        for s in self.shutter_speed:
            self.camera.set_shutter_speed(s) # one exposure
            raw_uint16_data = self.capture_uint16(slm_phase)
            raw_uint16_data_list.append(raw_uint16_data)
        #captured_intensity_hdr = self.process_raw_data(raw_uint16_data_list[0]) # convert to hdr and demosaick?
        captured_intensity_exposure_stack = [torch.clip(self.process_raw_data(raw_data), 0, 1) for raw_data in raw_uint16_data_list] # overexposed images, clip to range
        captured_intensity_hdr = self.merge_hdr(captured_intensity_exposure_stack)
        return captured_intensity_hdr, captured_intensity_exposure_stack

    def merge_hdr(self, exposure_stack):
        weight_sum = 0
        weighted_img_sum = 0
        for s, img in zip(self.shutter_speed, exposure_stack):
            weight = torch.exp(-4 * (img - 0.5)**2 / 0.5**2 )
            weighted_img = weight * (torch.log(img) - torch.log(torch.tensor(s)))
            weight_sum = weight_sum + weight
            weighted_img_sum = weighted_img_sum + weighted_img
        merged_img = torch.exp(weighted_img_sum / (weight_sum + 1e-10)) # numerical issues
        return merged_img

    def display_slm_phase(self, slm_phase):
        if slm_phase is not None: # just for simple camera capture
            if torch.is_tensor(slm_phase): # raw phase is always tensor. 
                slm_phase_encoded = phase_encoding(slm_phase, self.params_slm.slm_type)
            else: # uint8 encoded phase (should be np.array)
                slm_phase_encoded = slm_phase
            self.slm.updateArray(slm_phase_encoded)

    def capture_uint16(self):
        """
        gets phase pattern(s) and display it on the SLM, and then send a signal to board (wait next clock from SLM).
        Right after hearing back from the SLM, it sends another signal to PC so that PC retreives the camera buffer.

        :param slm_phase:
        :return:
        """

        if self.camera.params.ser is not None:
            self.camera.params.ser.write(f'D'.encode())

            # TODO: make the following in a separate function.
            # Wait until receiving signal from arduino
            incoming_byte = self.camera.params.ser.inWaiting()
            t0 = time.perf_counter()
            while True:
                received = self.camera.params.ser.read(incoming_byte).decode('UTF-8')
                if received != 'C':
                    incoming_byte = self.camera.params.ser.inWaiting()
                    if time.perf_counter() - t0 > 2.0:
                        break
                else:
                    break
        else:
            #print("settling...")
            time.sleep(self.params_slm.settle_time)
        raw_data_from_buffer = self.camera.retrieve_buffer()
        
        return raw_data_from_buffer

    def process_raw_data(self, raw_data):
        """
        gets raw data from the camera buffer, and demosaick it

        :param raw_data:
        :return:
        """
        raw_data = raw_data - 64
        color_cv_image = cv2.cvtColor(raw_data, self.camera.demosaick_rule)  # it gives float64 from uint16 -- double check it
        captured_intensity = utils.im2float(color_cv_image)  # float64 to float32

        # Numpy to tensor
        captured_intensity = torch.tensor(captured_intensity, dtype=torch.float32,
                                          device=self.dev).permute(2, 0, 1).unsqueeze(0)
        captured_intensity = torch.sum(captured_intensity, dim=1, keepdim=True)
        return captured_intensity

    def disconnect(self):
        #self.camera.stop_capture()
        self.camera.stop_capture()
        self.camera.disconnect()
        self.slm.close()

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.warper is not None:
            slf.warper = slf.warper.to(*args, **kwargs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf