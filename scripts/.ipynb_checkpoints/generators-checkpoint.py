class VLMGenerator(Dataset):
    def __init__(self, dataset_path, videos, model_type='BLIP2', size_t=364, 
                 video_dict=None, device=torch.device('cuda'),
                 num_segments=1, frames_per_segment=8):
        super(VideoGenerator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.model_type = model_type  # 'BLIP2' or 'CLIP'
        self.size_t = size_t
        self.video_dict = video_dict
        self.device = device
        self.num_segments = num_segments  # Number of segments to divide video into
        self.frames_per_segment = frames_per_segment
        
        # Normalization parameters for both models
        self.v_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,1,3)
        self.v_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,1,3)
        
        # Adjust size_t based on model type
        if model_type == 'CLIP':
            self.size_t = 288  # CLIP uses smaller size by default

    def __len__(self):
        return len(self.videos)

    def normalize(self, data):
        return (data / 255.0 - self.v_mean) / self.v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(364, 364), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube

    def segment_frames2tensor(self, frames, num_segments, frames_per_segment, target_size=(364, 364)):
        total_frames = len(frames)
        segments = []
        min_frames_for_average = frames_per_segment
    
        if total_frames >= num_segments * min_frames_for_average:
            # Average segment mode
            segment_length = total_frames // num_segments
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                segment_frames = frames[start:end]
                
                if len(segment_frames) > frames_per_segment:
                    step = max(1, len(segment_frames) // frames_per_segment)
                    segment_frames = segment_frames[::step][:frames_per_segment]
                elif len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))

                segment_frames = [cv2.resize(x[:, :, ::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                segments.append(vid_tube)
        else:
            # Sliding window mode
            step = max(1, (total_frames - frames_per_segment) // (num_segments - 1))
            for i in range(num_segments):
                start = min(i * step, total_frames - frames_per_segment)
                end = start + frames_per_segment
                segment_frames = frames[start:end]
                
                if len(segment_frames) < frames_per_segment:
                    segment_frames += [segment_frames[-1]] * (frames_per_segment - len(segment_frames))

                segment_frames = [cv2.resize(x[:, :, ::-1], target_size) for x in segment_frames]
                vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in segment_frames]
                vid_tube = np.concatenate(vid_tube, axis=1)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
                vid_tube = torch.from_numpy(vid_tube).to(self.device, non_blocking=True).float()
                segments.append(vid_tube)
        
        return torch.cat(segments, dim=0)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def __getitem__(self, idx):
        try:
            video = cv2.VideoCapture(self.video_dict[self.videos[idx]]['video_path'])
        except Exception as e:
            print('【Vid】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        
        frames = [x for x in self._frame_from_video(video)]
        frames_per_segment = min(self.frames_per_segment, len(frames))
        
        if len(frames) < self.num_segments * frames_per_segment or frames_per_segment == 0:
            print('【Len】')
            print(self.videos[idx])
            return None, None, self.videos[idx]
        
        if self.num_segments == 1:
            video_tensor = self.frames2tensor(
                frames, 
                fnum=frames_per_segment, 
                target_size=(self.size_t, self.size_t), 
                device=self.device
            )  
        else:
            video_tensor = self.segment_frames2tensor(
                frames,
                num_segments=self.num_segments,
                frames_per_segment=frames_per_segment,
                target_size=(self.size_t, self.size_t)
            )
            
        return video_tensor, self.video_dict[self.videos[idx]]['query_text'], self.videos[idx]
