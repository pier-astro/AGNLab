# Working with Redshift-Corrected Spectra and Instrumental Response

## The Problem

When you call `spec.zCorrect()`, the wavelength grid changes from observed frame to rest frame:
- `wave_obs` → `wave_rest = wave_obs / (1+z)`

But the **instrumental response matrix** is defined on the **observed wavelength grid** (the telescope's native grid). After `zCorrect()`, you lose access to `wave_obs` unless you save it manually.

## The Solution

The `Spectrum` class now automatically caches the observed-frame wavelength:

```python
spec.zCorrect()
# spec.wave is now REST FRAME
# spec.observed_wave is OBSERVED FRAME (automatically cached!)
```

## How It Works

### 1. Automatic Caching

```python
spec = Spectrum.from_txt('data.txt', z=0.1)

# Before zCorrect()
print(spec.wave[0])          # 6600.0 Å (observed)
print(spec.observed_wave[0]) # 6600.0 Å (same, returns copy)

# Apply redshift correction
spec.zCorrect()

# After zCorrect()
print(spec.wave[0])          # 6000.0 Å (rest frame)
print(spec.observed_wave[0]) # 6600.0 Å (cached observed frame!)
```

### 2. The `observed_wave` Property

```python
@property
def observed_wave(self):
    """Return wavelength in OBSERVED frame (before z-correction).
    
    - Before zCorrect(): Returns current wavelength
    - After zCorrect(): Returns cached observed wavelength
    """
```

## Correct Workflow

### Option 1: Explicit Wavelength

```python
# 1. Load spectrum
spec = Spectrum.from_txt('data.txt', z=0.1)

# 2. Crop/mask in observed frame
spec.crop(wbounds=(6000, 7000))

# 3. Apply corrections
spec.DeRedden()
spec.zCorrect()  # spec.wave → rest frame, observed_wave → cached

# 4. Create response using observed wavelengths
rsp = SpectralRsp.from_instrument('MUSE', wave=spec.observed_wave)

# 5. Fit
convolved = rsp(my_model)
spec.fit(convolved)
```

### Option 2: Pass Spectrum Directly (Recommended!)

```python
# 1-3. Same as above
spec = Spectrum.from_txt('data.txt', z=0.1)
spec.crop(wbounds=(6000, 7000))
spec.DeRedden()
spec.zCorrect()

# 4. Pass spectrum directly - automatic!
rsp = SpectralRsp.from_instrument('MUSE', spectrum=spec)
# Automatically uses spec.observed_wave

# 5. Fit
convolved = rsp(my_model)
spec.fit(convolved)
```

## What Happens During Fitting

```
┌─────────────────────────────────────────────────────────┐
│                    FITTING PROCESS                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Model evaluation (REST FRAME)                       │
│     model(spec.wave)  →  flux_rest                      │
│                                                          │
│  2. Transform to observed frame                         │
│     Using rsp.wave_grid = spec.observed_wave            │
│                                                          │
│  3. Apply instrumental response (OBSERVED FRAME)        │
│     response_matrix @ flux_rest  →  flux_convolved      │
│                                                          │
│  4. Compare to data (OBSERVED FRAME)                    │
│     chi2(flux_convolved, spec.flux_observed)            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

Wait, there's a subtlety here! After `zCorrect()`, `spec.flux` is also in the rest frame. Let me check the data comparison...

Actually, looking at the Sherpa fitting code:
```python
dataobj = Data1D("AGN", self.wave, self.flux, self.fluxerr)
```

After `zCorrect()`, both `self.wave` and `self.flux` are in the rest frame. The model should also evaluate in the rest frame and return rest-frame flux. So the instrumental response needs to map from rest frame to rest frame... but wait, that doesn't make sense either!

## The Real Issue

Let me think about this more carefully:

1. **Observed data**: `(λ_obs, F_obs)` - This is what comes from the telescope
2. **After zCorrect()**: `(λ_rest, F_rest)` where:
   - `λ_rest = λ_obs / (1+z)`
   - `F_rest = F_obs × (1+z)`

3. **Instrumental response matrix**: Operates on the native telescope grid (observed frame)

The confusion arises because:
- The **instrumental convolution happens in the observed frame** (λ_obs grid)
- But after `zCorrect()`, we're **fitting in the rest frame** (λ_rest grid)

The response matrix should be:
- Built on the λ_obs grid
- But applied to models evaluated on the λ_rest grid
- The ConvolvedModel needs to handle the frame transformation

Let me check the ConvolvedModel.calc() to see how it handles this...

Actually, I think the issue is that the **response matrix should be applied in whatever frame the data is in**. If you call `zCorrect()`, you've moved everything to the rest frame, so the response should also be in the rest frame.

But that's wrong! The instrumental response is a physical property of the telescope - it doesn't care about redshift!

## The Correct Understanding

The instrumental response matrix describes how the instrument spreads light **in the observed frame**. When you do `zCorrect()`:

1. You transform wavelengths and fluxes to rest frame
2. But the instrumental **resolution** also changes with wavelength
3. The response matrix needs to be built on the **observed** wavelength grid
4. But then **rescaled** to the rest frame

So the correct approach is:
1. Build response on `observed_wave` grid
2. When model evaluates on `rest_wave` = `observed_wave / (1+z)`
3. The response automatically scales because the wavelength grid changes

Let me update the documentation to reflect this correct understanding...

Actually, looking at the code more carefully, I think the current implementation is correct. The `wave_grid` parameter to SpectralRsp should be the wavelength grid that corresponds to the data you're fitting. If you've called `zCorrect()`, that grid is in the rest frame, but you need to provide the **observed frame** grid so the response matrix is built correctly.

Let me simplify the documentation to be clearer:

