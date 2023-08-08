"""Basic feature pipelines

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Sarthak Yadav 2020
"""
import torch
from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
    Filterbank,
    DCT,
    Deltas,
    ContextWindow,
)
from speechbrain.nnet.CNN import GaborConv1d
from speechbrain.nnet.normalization import PCEN
from speechbrain.nnet.pooling import GaussianLowpassPooling
#from speechbrain.lobes.chroma_filters import get_chroma_filters #alan version
from speechbrain.lobes.tina_chroma_filters import get_chroma_filters
from speechbrain.lobes.data_preprocessing import get_lpfilter
#from speechbrain.lobes.data_preprocessing import get_chroma_filters, get_lpfilter #alan version
#import torchaudio
import math

class Chroma(torch.nn.Module):
    def __init__(
            self,
            n_chroma = 48,  #smaller or equal to 48, different from n_mels when use_cor = True
            win_length = 129, #kernel_size
            hop_length=64, #stride between 1 and win_length, only if use_cor=False
            chroma_coef=15,
            padding=0, #only if use_cor=False
            lpf = False,
            use_cor = False,
    ):
        super(Chroma,self).__init__()
        
        #print("memory cuda", torch.cuda.memory_allocated(device='cuda'))
        self.n_chroma = n_chroma
        self.cor = use_cor
        if self.cor :
            self.cf = torch.nn.Conv1d(in_channels=1,out_channels=n_chroma,kernel_size=win_length,stride=1,padding="same",bias=False).cuda()

        else :  
            self.cf = torch.nn.Conv1d(in_channels=1,out_channels=n_chroma,kernel_size=win_length,stride=hop_length,padding=padding,bias=False).cuda()
        
        #print("memory cuda after conv1d init", torch.cuda.memory_allocated(device='cuda'))


        CF = get_chroma_filters().cuda()
        CF = CF.flip(1) #added by louis
        #print("Chroma.__init__()")
        #print(self.cf.weight.data.size(),CF.size())
        CF = CF[:n_chroma, :]
        self.cf.weight.data = CF.unsqueeze(1)*10**(-1*chroma_coef)
        #print(torch.sum(self.cf.weight.data*self.cf.weight.data))
        

        self.lpf = lpf
        
        if self.lpf:
            print("Init lpf")
            ker_size = CF.size()[0] #129, 130?
            self.conv_lpf = torch.nn.Conv1d(in_channels=1,out_channels=1,kernel_size=ker_size,stride=1,padding='same',bias=False).cuda()
            filt = get_lpfilter().cuda()
            filt = filt.unsqueeze(0)
            filt = filt.unsqueeze(0)
            self.conv_lpf.weight.data = filt.flip(2)

            #print("memory cuda after lpf init", torch.cuda.memory_allocated(device='cuda'))


    def forward(self,wav):
        wav = wav.unsqueeze(1)
        #print("wav max", wav.max(), "min", wav.min(), "std", wav.std())
        #wav = wav * 2**15 #unnormalization
        #print("unnorm wav max", wav.max(), "min", wav.min(), "std", wav.std())


        n_chroma = self.n_chroma
        #print(n_chroma)

        if self.lpf : #low pass filter
            #print("wav before lpf", wav.size())
            wav = self.conv_lpf(wav).detach()
            #print("wav after lpf", wav.size())

            #print("memory cuda after applying lpf", torch.cuda.memory_allocated(device='cuda'))


        if self.cor :
            chroma_feat = self.cf(wav).detach()
            #print("chroma feat size bf transpose", chroma_feat.size())
            chroma_feat = chroma_feat.transpose(1,2) 
            
            #print("nan in chroma feat ?", torch.any(torch.isnan(chroma_feat)), torch.numel(chroma_feat), torch.isnan(chroma_feat).sum())
            #print("chroma feat exp", chroma_feat[0,:5,:3], chroma_feat[0,1000:1010,:3], "max", chroma_feat.max(), "min", chroma_feat.min())

            win_cor = 25e-3 #e.g. 25ms
            hop_cor = 10e-3 #e.g. 10ms
            fs = 16e3 #sampling freq 16kHz
            win_index = int(win_cor*fs)
            hop_index = int (hop_cor*fs)
            beg = 0
            end = beg + win_index
            length_sig = chroma_feat.size()[1]
            #print("len sig/chroma", chroma_feat.size())
            size_out = math.ceil(length_sig/hop_index)  
            #features = torch.zeros(16, size_out, 48*48).cuda()
            

            idx = torch.triu_indices(n_chroma,n_chroma,1)#only keep upper triangular coefficients + no diagonal coef (offset=1) (total : 48*47/2=1128)
            
            #idx = [[],[]]
            #for i in range(n_chroma):
            #    for j in range(n_chroma):
            #        idx[0].append(i)
            #        idx[1].append(j)
            #idx = torch.tensor(idx)



            #remove the zeroes (i+j odd)
            #idx = [[],[]]
            #for i in range(n_chroma):
            #    for j in range(n_chroma):
            #        if ((i+j)%2==0) and (j>=i): #j>i if no diagonal coef
            #            idx[0].append(i)
            #            idx[1].append(j)
            #idx = torch.tensor(idx)
            
            #even1, even2 = [], []
            #odd1, odd2 = [], []
            #idx = [[],[]]
            #for i in range(n_chroma):
            #    for j in range(n_chroma):
            #        if ((i+j)%2==0) and (j>i):
            #            if i%2==0:
            #                even1.append(i)
            #                even2.append(j)
            #            else :
            #                odd1.append(i)
            #                odd2.append(j)
            #idx = torch.tensor([even1 + odd1, even2 + odd2])


            features = torch.zeros(wav.size()[0], size_out, int(n_chroma*(n_chroma-1)/2)).cuda() #(16,size_out,48*47//2)
            #features = torch.zeros(wav.size()[0], size_out, int((n_chroma/2)*(n_chroma/2-1))).cuda() #if no diago
            #features = torch.zeros(wav.size()[0], size_out, int((n_chroma/2)*(n_chroma/2+1))).cuda()
            #features = torch.zeros(wav.size()[0], size_out, int(n_chroma*(n_chroma+1)/2)).cuda()
            #features = torch.zeros(wav.size()[0], size_out, int(n_chroma**2)).cuda()


            #print("expected feature size", features.size())
            #for k in range(16) :
            #print("memory cuda after k change", torch.cuda.memory_allocated(device='cuda'))

            #print('k', k)
            c = 0
            # ??  input_shape # (16, 134, 48)
            beg = 0
            while beg < length_sig:
                #print('beg', beg, 'len sig', length_sig)
                if end > length_sig :
                    enf = length_sig
                
                #partial_chroma = chroma_feat[k,beg:end,:].squeeze(0) #.cuda() ?
                
                batchPChroma = chroma_feat[:,beg:end,:].transpose(1,2)
                mpc = torch.mean(batchPChroma, 2)
                bpc = batchPChroma.sub(mpc.unsqueeze(2).expand_as(batchPChroma)) #substact the mean
                cov = torch.bmm(bpc, bpc.transpose(1,2))
                cov = cov / (batchPChroma.size()[1]-1) #unbiased covariance
                #print("nan in cov?", torch.any(torch.isnan(cov)), torch.numel(cov), torch.isnan(cov).sum())

                d = torch.diagonal(cov, dim1=1, dim2=2)
                std = torch.pow(d, 0.5).unsqueeze(2).expand_as(cov)
                
                #first version

                #print("nan in std?", torch.any(torch.isnan(std)), torch.numel(std), torch.isnan(std).sum())
                #print("check if 0 in std", std.all(), std.min())
                cov[cov==0] = 1e-10#replace 0 with very small value to avoid nan
                cor = cov.div(std) #+0.5??
                cor = cor.div(std.transpose(1,2))
                cor = cor #*0.5  + 0.5 #Tina regularization ?
                #print("nan in cor?", torch.any(torch.isnan(cor)), torch.numel(cor), torch.isnan(cor).sum())
                #print(cor.min(), cor.max())
                cor = torch.clamp(cor, -1.0, 1.0) #inf -> 1
                
                #normalization
                #cor = torch.nn.functional.normalize(cor)

                #second version with log
                #std2 = std.mul(std.transpose(1,2))
                #cov[cov==0] = 1e-10#replace 0 with very small value to avoid nan
                #cor = cov.div(std2) #+0.5??
                #logstd2 = torch.log(std2+1)
                #cor = cor.mul(logstd2)


                #cov = torch.matmul(partial_chroma.transpose(0,1), partial_chroma) #.cuda() ?
                #cov = torch.bmm(chroma_feat[:,beg:end,:].transpose(1,2), chroma_feat[:,beg:end,:]) #dim (16,48,48)
                #cov = torch.cov(partial_chroma)
                #cor = torch.zeros_like(cov) #.cuda() ?
                #cor = torch.zeros(wav.size()[0],n_chroma,n_chroma) #e.g (16,48,48)

                #print("cor, cov size", cor.size(), cov.size())
                #for i in range(cov.size()[0]): #48
                #    for j in range(cov.size()[1]): #48
                #        cor[i,j] = 0.5 * cov[i,j] / torch.sqrt(cov[i,i] * cov[j,j]) + 0.5
                    
                for k in range(cor.size()[0]): #16
                    #print("k", k, "nan in chroma_feat?", torch.any(torch.isnan(chroma_feat[k,beg:end,:])), torch.numel(chroma_feat[k,beg:end,:]), torch.isnan(chroma_feat[k,beg:end,:]).sum())
                    #print("k", k, "inf in chroma_feat?", torch.any(torch.isinf(chroma_feat[k,beg:end,:])), torch.numel(chroma_feat[k,beg:end,:]), torch.isinf(chroma_feat[k,beg:end,:]).sum())
                    #print("chroma_fit first coef", chroma_feat[k:beg:beg+5, :5])
                    
                    #cor[k, :, :] = torch.corrcoef(chroma_feat[k,beg:end,:].squeeze(0).transpose(0,1))
                    
                    #print("chroma_feat", chroma_feat.size(), beg, end, chroma_feat[k,beg:end,:].size())
                    #print("size before cor", chroma_feat[k,beg:end,:].squeeze(0).transpose(0,1).size())
                    #print("k", k, "nan in cor?", torch.any(torch.isnan(cor[k, :, :])), torch.numel(cor[k, :, :]), torch.isnan(cor[k, :, :]).sum())
                    
                    

                    #print("c", c , "k", k )
                    #for i in range(cor.size()[1]): #48
                        #for j in range(cor.size()[2]): #48
                            #cor[k,i,j] = 0.5 * cov[k,i,j] / torch.sqrt(cov[k,i,i] * cov[k,j,j]) + 0.5
                    
                    

                    #idx = torch.triu_indices(n_chroma,n_chroma,1)#only keep upper triangular coefficients + no diagonal coef (=1) (total : 48*47/2=1128)
                    features[k, c, :] = cor[k, idx[0], idx[1]]
                    #print("nan in features?", torch.any(torch.isnan(cor)), torch.numel(cor), torch.isnan(cor).sum())
    

                #print("size feature", features[k, c, :].size(), torch.flatten(cor).size())
                    
                #idx = torch.triu_indices(48,48,1)#only keep upper triangular coefficients + no diagonal coef (=1) (total : 48*47/2=1128)
                #features[k, c, :] = cor[idx[0], idx[1]]
                   
                #features[k, c, :] = torch.flatten(cor)
                #print('c', c)
                c+=1
                #print("memory cuda after c change", torch.cuda.memory_allocated(device='cuda'))

                beg += hop_index    
                end += hop_index
            #print('features before transpose', features.size())
            #features = features.transpose(1,2)
            
            #check if Nan ?
            print("nan ?", torch.any(torch.isnan(features)), torch.numel(features), torch.isnan(features).sum())
            features[features != features] = 0
            #print("inf ?", torch.any(torch.isinf(features)))
                                    

        else :
            #print("Chroma.forward")
            features = self.cf(wav).detach() #no need for gradient
            features = features.transpose(1,2)
            
        #normalizing ?
        #print("memory cuda before returning features", torch.cuda.memory_allocated(device='cuda'))

        #print("input",wav.size(),"features",features.size(), "1st coefs", features[:1,:5,:3])
        #print("min", features.min(), "max", features.max(), "mean", features.mean(), "median", features.median(), "std", features.std())

        return features

class Fbank(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: False)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: False)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 16000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 40)
        Number of Mel filters.
    filter_shape : str (default: triangular)
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor : float (default: 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor : float (default: 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default: 5)
        Number of frames of left context to add.
    right_frames : int (default: 5)
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Fbank()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        deltas=False,
        context=False,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=40,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        left_frames=5,
        right_frames=5,
        win_length=25,
        hop_length=10,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_deltas = Deltas(input_size=n_mels)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )

    def forward(self, wav):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag)
        if self.deltas:
            delta1 = self.compute_deltas(fbanks)
            delta2 = self.compute_deltas(delta1)
            fbanks = torch.cat([fbanks, delta1, delta2], dim=2)
        if self.context:
            fbanks = self.context_window(fbanks)
        
        #print("wav input",wav.size(),"fbanks",fbanks.size(), "1st coefs", fbanks[:3,100:110,:3])
        #print("min", fbanks.min(), "max", fbanks.max(), "mean", fbanks.mean(), "median", fbanks.median(), "std", fbanks.std())
        return fbanks


class MFCC(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: True)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: True)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 16000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 23)
        Number of filters to use for creating filterbank.
    n_mfcc : int (default: 20)
        Number of output coefficients
    filter_shape : str (default 'triangular')
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor: bool (default 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor: float (default 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default 5)
        Number of frames of left context to add.
    right_frames : int (default 5)
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = MFCC()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 660])
    """

    def __init__(
        self,
        deltas=True,
        context=True,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=23,
        n_mfcc=20,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        left_frames=5,
        right_frames=5,
        win_length=25,
        hop_length=10,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_dct = DCT(input_size=n_mels, n_out=n_mfcc)
        self.compute_deltas = Deltas(input_size=n_mfcc)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )

    def forward(self, wav):
        """Returns a set of mfccs generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag)
        mfccs = self.compute_dct(fbanks)
        if self.deltas:
            delta1 = self.compute_deltas(mfccs)
            delta2 = self.compute_deltas(delta1)
            mfccs = torch.cat([mfccs, delta1, delta2], dim=2)
        if self.context:
            mfccs = self.context_window(mfccs)
        return mfccs


class Leaf(torch.nn.Module):
    """
    This class implements the LEAF audio frontend from

    Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc. of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    window_len: float
        length of filter window in milliseconds
    window_stride : float
        Stride factor of the filters in milliseconds
    sample_rate : int,
        Sampling rate of the input signals. It is only used for sinc_conv.
    min_freq : float
        Lowest possible frequency (in Hz) for a filter
    max_freq : float
        Highest possible frequency (in Hz) for a filter
    use_pcen: bool
        If True (default), a per-channel energy normalization layer is used
    learnable_pcen: bool:
        If True (default), the per-channel energy normalization layer is learnable
    use_legacy_complex: bool
        If False, torch.complex64 data type is used for gabor impulse responses
        If True, computation is performed on two real-valued tensors
    skip_transpose: bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 8000])
    >>> leaf = Leaf(
    ...     out_channels=40, window_len=25., window_stride=10., in_channels=1
    ... )
    >>> out_tensor = leaf(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        out_channels,
        window_len: float = 25.0,
        window_stride: float = 10.0,
        sample_rate: int = 16000,
        input_shape=None,
        in_channels=None,
        min_freq=60.0,
        max_freq=None,
        use_pcen=True,
        learnable_pcen=True,
        use_legacy_complex=False,
        skip_transpose=False,
        n_fft=512,
    ):
        super(Leaf, self).__init__()
        self.out_channels = out_channels
        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.complex_conv = GaborConv1d(
            out_channels=2 * out_channels,
            in_channels=in_channels,
            kernel_size=window_size,
            stride=1,
            padding="same",
            bias=False,
            n_fft=n_fft,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            use_legacy_complex=use_legacy_complex,
            skip_transpose=True,
        )

        self.pooling = GaussianLowpassPooling(
            in_channels=self.out_channels,
            kernel_size=window_size,
            stride=window_stride,
            skip_transpose=True,
        )
        if use_pcen:
            self.compression = PCEN(
                self.out_channels,
                alpha=0.96,
                smooth_coef=0.04,
                delta=2.0,
                floor=1e-12,
                trainable=learnable_pcen,
                per_channel_smooth_coef=True,
                skip_transpose=True,
            )
        else:
            self.compression = None
        self.skip_transpose = skip_transpose

    def forward(self, x):
        """
        Returns the learned LEAF features

        Arguments
        ---------
        x : torch.Tensor of shape (batch, time, 1) or (batch, time)
            batch of input signals. 2d or 3d tensors are expected.
        """

        if not self.skip_transpose:
            x = x.transpose(1, -1)

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        outputs = self.complex_conv(x)
        outputs = self._squared_modulus_activation(outputs)
        outputs = self.pooling(outputs)
        outputs = torch.maximum(
            outputs, torch.tensor(1e-5, device=outputs.device)
        )
        if self.compression:
            outputs = self.compression(outputs)
        if not self.skip_transpose:
            outputs = outputs.transpose(1, -1)
        return outputs

    def _squared_modulus_activation(self, x):
        x = x.transpose(1, 2)
        output = 2 * torch.nn.functional.avg_pool1d(
            x ** 2.0, kernel_size=2, stride=2
        )
        output = output.transpose(1, 2)
        return output

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels.
        """

        if len(shape) == 2:
            in_channels = 1
        elif len(shape) == 3:
            in_channels = 1
        else:
            raise ValueError(
                "Leaf expects 2d or 3d inputs. Got " + str(len(shape))
            )
        return in_channels
