# Here, we explain the provided forward-backward MDP problems:

## EdgeCaching.py
This file implements an instance of a forward-backward Markov decision process (FB-MDP) in the domain of wireless caching. 
the environment entails cache-equipped serving nodes (SNs) distributed spatially with intensities $\lambda_{\rm sn}$. Each SN multicasts popular contents to users, at the start of each time-slot, aiming to satisfy as many users as possible. The transmission at time-slot $t$ will be completed within a duration of $d(t)$ seconds. Requests for contents can span multiple time slots until successfully fulfilled, influencing expected latencies for content reception. The problem is modeled as a FB-MDP, coupling forward and backward dynamics through system actions. The action space is $[0,1]^N\times[0,\infty]^N$ with $N$ the number of contents. The environment metrics include quality-of-service $r_{\rm QoS}$ and total bandwidth consumption $r_{\rm BW}$, as forward rewards, as well as  overall expected latency $r_{\rm Lat}$, as a backward reward, which are leveraged to guide the policy optimization. While $r_{\rm QoS}$ and $r_{\rm BW}$ relate to forward states, $r_{\rm Lat}$ interacts with both forward and backward states. Note that  $r_{\rm QoS}$ determines the overall probability of unsatisfied UEs, $r_{\rm BW}$ measures the total bandwidth consumption of the network and $r_{\rm Lat}$ determines the expected latency for successful reception of contents.
This setup then presents a multi-objective FB-MDP problem, which is solved by FB-MOAC algorithm. 

### Hyper-parameters: 
There are various hyper-parameters related to the system model of the considered environment that can be set in this code as follows:

        self.Nfile            = 200                      # number of contents to be multicasted
        self.Skewness         = 0.6                      # skewness of content popularity
        self.StateDim_FW      = self.Nfile               # dimension of forward state
        self.StateDim_BW      = self.Nfile               # dimension of backward state
        self.CacheCapacity    = 10                       # cache capacity of serving nodes
        self.ActionDim        = 2*self.Nfile+1           # dimension of action
        self.CoupledStateDim  = self.Nfile               # dimension of variables coupled between forward and backward states
        self.N_forwadRewards  = 2                        # number of forward reward functions
        self.N_backwadRewards = 1                        # number of backward reward functions

       
        self.Lambda_UE_fixed     = 1e5                      # Spatial intensity of users
        self.Lambda_HN           = 100                      # Spatial intensity of serving nodes
        self.Rate                = 1e3                      # Desired information rate of contents
        self.P_N0_               = 2e7                      # Ratio between transmitting power of serving nodes to the spectral noise
        self.FileDuration        = 600.0                    # Slot duration in seconds
        self.NormalizationFactor = 1000
        self.gammaR = self.P_N0_/self.Rate/self.NormalizationFactor             


        self.CacheWeight_old = [[np.zeros(self.Nfile)]]     # Cache probability of serving nodes
        self.Lambda_UE_n = np.zeros([self.Nfile])           # Spatial intensity of "requesting" users
        self.Total_Outage = np.zeros([self.Nfile])          # Instantaneous total outage of content transmissions

        # Harmonic numbers of the harmonic broadcasting to effeciently decrease the latency
        self.invLatencyRelatedToHarmonic = np.array([1.0, 4.0, 11.0, 30.0, 83.0, 227.0, 620.0, 1680.0, 4550.0, 12400.0])
        self.Latency = self.FileDuration/self.invLatencyRelatedToHarmonic  # Instantaneous latency for different tasks            

### Outputs: 
*Forward State*:  content-specific intensity of requesting users from the network.

*Forward Rewards*: Quality-of-Service $r_{QoS}$ and bandwidth consumption $r_{BW}$.

*Backward State*:  content-specific experinece delay.

*Backward Rewards*: Expected latency $r_{Lat}$.






------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## ComputationOffloading.py

This experiment delves into the realm of computation offloading within a network comprising multiple mobile devices and a centralized edge server. The setup involves $N_{dev}$ mobile devices and $N_t$ computational tasks with varying sizes. Each mobile device exhibits preferences for specific tasks, determining the likelihood of offloading them to the edge server. The server operates in discrete time slots $t$, each of duration $\tau$ seconds, and allocates buffer capacity and computational resources to handle the offloaded tasks. Importantly, the system accounts for the possibility of buffer overflow, necessitating task re-offloading. The average computation time for a task hinges on whether it's processed locally or offloaded, influenced by factors such as queue length and resource allocation. The experiment seeks to optimize two crucial objectives: minimizing overflow probability $r_{OP}$ and expected computation time $r_{CT}$. These objectives present a trade-off, constituting a feedback-controlled Markov decision process (FB-MDP) environment with a multidimensional action space.


### Hyper-parameters: 
There are various hyper-parameters related to the system model of the considered environment that can be set in this code as follows:

        self.env_name = 'ComputationOffloading'
        
        self.Nfile    = 100                                          # number of tasks
        self.N_UE     = 20                                           # number of users offloading their computations
        self.FileSize = 10.0 * np.ones([self.Nfile])            
        self.FileSize = np.linspace(1, 100, self.Nfile) + 10.0       # tasks size in [KBits]
        self.N_UE_times_Filesize = self.N_UE * self.FileSize
        self.Skewness = 0.6                                          # skewness of tasks popularity
        
        self.StateDim_FW     = 2*self.Nfile                          # dimension of forward state
        self.StateDim_BW     = 1*self.Nfile                          # dimension of backward state
        self.ActionDim       = 3*self.Nfile                          # dimension of action state
        self.CoupledStateDim = 2*self.Nfile                          # dimension of variables coupled between forward and backward states
        self.N_forwadRewards  = 1                                    # number of forward reward functions
        self.N_backwadRewards = 1                                    # number of backward reward functions

        self.ResourceComputationUE = 10                              # computation resource of each user in [Kbits/slot]
        self.BufferCapacity        = 100                             # buffer capacity of the cloud [Kbits]
        self.ComputationCapacity   = 100                             # computation resource of the cloud in [Kbits/slot], should be greater than that of users
        
        
        self.SlotDuration     = 60.0                                   # Slot duration in [Seconds]
        self.QueueLength      = np.zeros([self.Nfile])                 # Queue length of different buffers in [KBits]
        self.Outage           = np.zeros([self.Nfile])                 # Overflow probability of different buffer [0,1]
        self.OffloadProb      = np.zeros([self.Nfile])                 # Offloading probability of different tasks
 

### Outputs: 
*Forward State*:  content-specific preference of tasks and content-specific queue lengths of buffers.

*Forward Rewards*: overflow probability $r_{OP}$.

*Backward State*:  content-specific experinece delay.

*Backward Rewards*: expected computation time $r_{CT}$.

