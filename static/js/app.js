/**
 * static/js/app.js
 * Client application controller for Thai License Plate Recognition Dashboard.
 * Handles:
 *  - Mode switching (Upload vs Live RTSP Stream)
 *  - Debug mode toggling (ON/OFF)
 *  - Drag & drop single/batch image and video upload
 *  - Batch carousel navigation
 *  - 3-stage visual pipeline breakdown rendering
 *  - Debug breakdown drawer with province probability distribution
 *  - Live MJPEG stream connection and real-time detection polling
 */

(function () {
  'use strict';

  // --- State Management ---
  const state = {
    mode: 'upload', // 'upload' | 'live'
    isDebug: false,
    batchResults: [],
    currentIndex: 0,
    isStreaming: false,
    streamPollTimer: null,
  };

  // --- DOM Elements ---
  const tabUpload = document.getElementById('tabUpload');
  const tabLive = document.getElementById('tabLive');
  const uploadPanel = document.getElementById('uploadPanel');
  const livePanel = document.getElementById('livePanel');
  const debugToggle = document.getElementById('debugToggle');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const uploadCountPill = document.getElementById('uploadCountPill');
  const batchGalleryWrapper = document.getElementById('batchGalleryWrapper');
  const batchGallery = document.getElementById('batchGallery');

  // RTSP Elements
  const rtspInput = document.getElementById('rtspInput');
  const btnConnectStream = document.getElementById('btnConnectStream');
  const streamViewer = document.getElementById('streamViewer');
  const streamPlaceholder = document.getElementById('streamPlaceholder');
  const streamStatusPill = document.getElementById('streamStatusPill');

  // Pipeline Breakdown Elements
  const totalLatencyPill = document.getElementById('totalLatencyPill');
  
  // Stage 0: Raw
  const timeRaw = document.getElementById('timeRaw');
  const cropRaw = document.getElementById('cropRaw');
  const cropRawPlaceholder = document.getElementById('cropRawPlaceholder');
  const metaRes = document.getElementById('metaRes');
  const metaStatus = document.getElementById('metaStatus');

  // Stage 1: Model 1
  const timeM1 = document.getElementById('timeM1');
  const cropM1 = document.getElementById('cropM1');
  const cropM1Placeholder = document.getElementById('cropM1Placeholder');
  const confPlate = document.getElementById('confPlate');

  // Stage 2: Model 2
  const timeM2 = document.getElementById('timeM2');
  const cropChar = document.getElementById('cropChar');
  const cropProv = document.getElementById('cropProv');
  const confChar = document.getElementById('confChar');
  const confProv = document.getElementById('confProv');

  // Stage 3: Model 3
  const timeM3 = document.getElementById('timeM3');
  const resultPlate = document.getElementById('resultPlate');
  const resultProvince = document.getElementById('resultProvince');
  const resultBadge = document.getElementById('resultBadge');
  const patternText = document.getElementById('patternText');
  const confProvProb = document.getElementById('confProvProb');

  // Debug Drawer
  const debugDrawer = document.getElementById('debugDrawer');
  const dbgPoly = document.getElementById('dbgPoly');
  const dbgDeskew = document.getElementById('dbgDeskew');
  const dbgComp = document.getElementById('dbgComp');
  const dbgOcr = document.getElementById('dbgOcr');
  const dbgProvBars = document.getElementById('dbgProvBars');

  // --- Initialization ---
  function init() {
    setupModeTabs();
    setupDebugToggle();
    setupDropzone();
    setupRTSPStream();
  }

  // --- Mode Switching ---
  function setupModeTabs() {
    tabUpload.addEventListener('click', () => {
      state.mode = 'upload';
      tabUpload.classList.add('active');
      tabLive.classList.remove('active');
      uploadPanel.style.display = 'block';
      livePanel.style.display = 'none';
      if (state.isStreaming) stopStream();
    });

    tabLive.addEventListener('click', () => {
      state.mode = 'live';
      tabLive.classList.add('active');
      tabUpload.classList.remove('active');
      uploadPanel.style.display = 'none';
      livePanel.style.display = 'block';
    });
  }

  // --- Debug Mode Toggle ---
  function setupDebugToggle() {
    debugToggle.addEventListener('change', (e) => {
      state.isDebug = e.target.checked;
      console.log(`[Debug Mode] Toggled: ${state.isDebug ? 'ON' : 'OFF'}`);

      // Re-render current detection drawer state
      const current = state.batchResults[state.currentIndex];
      if (current) {
        renderDebugDrawer(current);
      }
    });
  }

  // --- Drag & Drop Setup ---
  function setupDropzone() {
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleSelectedFiles(e.dataTransfer.files);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleSelectedFiles(e.target.files);
      }
    });
  }

  // --- Handle Files (Single/Batch Images or Video) ---
  async function handleSelectedFiles(fileList) {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    // Check if uploaded file is a video
    const isVideo = files[0].type.startsWith('video/') || files[0].name.match(/\.(mp4|mov|avi|mkv)$/i);

    if (isVideo) {
      await uploadVideoFile(files[0]);
    } else {
      await uploadImageFiles(files);
    }
  }

  // Upload Images (Batch or Single)
  async function uploadImageFiles(files) {
    uploadCountPill.textContent = `${files.length} file${files.length > 1 ? 's' : ''}`;
    metaStatus.textContent = 'Processing...';
    metaStatus.style.color = 'var(--accent-amber)';

    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    formData.append('debug', state.isDebug ? 'true' : 'false');
    formData.append('conf_m1', '0.35');
    formData.append('conf_m2', '0.25');

    try {
      const resp = await fetch('/api/detect/image', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) throw new Error(`Server returned HTTP ${resp.status}`);
      const data = await resp.json();

      state.batchResults = data.results || [];
      state.currentIndex = 0;

      if (state.batchResults.length > 1) {
        renderBatchGallery();
      } else {
        batchGalleryWrapper.style.display = 'none';
      }

      if (state.batchResults.length > 0) {
        renderPipelineResult(state.batchResults[0]);
      }
    } catch (err) {
      console.error('[Upload Error]', err);
      metaStatus.textContent = 'Inference Error';
      metaStatus.style.color = 'var(--accent-red)';
      alert(`Inference failed: ${err.message}`);
    }
  }

  // Upload Video File
  async function uploadVideoFile(videoFile) {
    uploadCountPill.textContent = '1 video';
    metaStatus.textContent = 'Processing Video...';
    metaStatus.style.color = 'var(--accent-amber)';

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('debug', state.isDebug ? 'true' : 'false');
    formData.append('sample_rate', '5');

    try {
      const resp = await fetch('/api/detect/video', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) throw new Error(`Server returned HTTP ${resp.status}`);
      const data = await resp.json();

      state.batchResults = data.results || [];
      state.currentIndex = 0;

      if (state.batchResults.length > 1) {
        renderBatchGallery(true);
      } else {
        batchGalleryWrapper.style.display = 'none';
      }

      if (state.batchResults.length > 0) {
        renderPipelineResult(state.batchResults[0]);
      } else {
        metaStatus.textContent = 'No plates detected in video';
        metaStatus.style.color = 'var(--accent-red)';
      }
    } catch (err) {
      console.error('[Video Error]', err);
      metaStatus.textContent = 'Video Error';
      metaStatus.style.color = 'var(--accent-red)';
      alert(`Video processing failed: ${err.message}`);
    }
  }

  // --- Render Batch Carousel ---
  function renderBatchGallery(isVideo = false) {
    batchGalleryWrapper.style.display = 'flex';
    batchGallery.innerHTML = '';

    state.batchResults.forEach((res, idx) => {
      const thumb = document.createElement('img');
      thumb.className = `batch-thumb ${idx === state.currentIndex ? 'active' : ''}`;
      thumb.src = (res.crops && (res.crops.plate_rectified || res.crops.raw)) || '';
      thumb.title = isVideo
        ? `Time: ${res.timestamp_sec}s | ${res.plate_text || 'No plate'}`
        : `${res.filename || 'Image ' + (idx + 1)} | ${res.plate_text || 'No plate'}`;

      thumb.addEventListener('click', () => {
        state.currentIndex = idx;
        document.querySelectorAll('.batch-thumb').forEach((el) => el.classList.remove('active'));
        thumb.classList.add('active');
        renderPipelineResult(res);
      });

      batchGallery.appendChild(thumb);
    });
  }

  // --- Render 3-Stage Pipeline Breakdown ---
  function renderPipelineResult(res) {
    if (!res) return;

    // Total Latency
    const totalMs = res.timing ? res.timing.total_ms : '--';
    totalLatencyPill.textContent = `Latency: ${totalMs} ms`;

    // Stage 0: Raw
    if (res.crops && res.crops.raw) {
      cropRaw.src = res.crops.raw;
      cropRaw.style.display = 'block';
      cropRawPlaceholder.style.display = 'none';
    } else {
      cropRaw.style.display = 'none';
      cropRawPlaceholder.style.display = 'flex';
    }
    timeRaw.textContent = `${totalMs} ms`;
    metaStatus.textContent = res.detected ? 'Plate Detected' : 'No Target';
    metaStatus.style.color = res.detected ? 'var(--accent-green)' : 'var(--accent-red)';
    metaRes.textContent = res.detected ? '320x160 Warp' : '--';

    // If detection failed
    if (!res.detected) {
      cropM1.style.display = 'none';
      cropM1Placeholder.style.display = 'flex';
      cropChar.style.display = 'none';
      cropProv.style.display = 'none';
      resultPlate.textContent = 'NO PLATE';
      resultProvince.textContent = 'None';
      resultBadge.textContent = 'NOT DETECTED';
      resultBadge.style.color = 'var(--accent-red)';
      patternText.textContent = '--';
      confPlate.textContent = '--';
      confChar.textContent = '--';
      confProv.textContent = '--';
      confProvProb.textContent = '--';
      debugDrawer.classList.remove('active');
      return;
    }

    // Stage 1: Model 1
    if (res.crops && res.crops.plate_rectified) {
      cropM1.src = res.crops.plate_rectified;
      cropM1.style.display = 'block';
      cropM1Placeholder.style.display = 'none';
    }
    timeM1.textContent = `${res.timing.m1_ms} ms`;
    confPlate.textContent = `${(res.confidence.plate_detection * 100).toFixed(1)}%`;

    // Stage 2: Model 2
    if (res.crops && res.crops.char_crop) {
      cropChar.src = res.crops.char_crop;
      cropChar.style.display = 'block';
    }
    if (res.crops && res.crops.prov_crop) {
      cropProv.src = res.crops.prov_crop;
      cropProv.style.display = 'block';
    }
    timeM2.textContent = `${res.timing.m2_ms} ms`;
    confChar.textContent = `${(res.confidence.char_detection * 100).toFixed(1)}%`;
    confProv.textContent = `${(res.confidence.prov_detection * 100).toFixed(1)}%`;

    // Stage 3: Model 3
    timeM3.textContent = `${res.timing.m3_ms} ms`;
    resultPlate.textContent = res.plate_text || '--';
    resultProvince.textContent = res.province || '--';

    if (res.is_valid) {
      resultBadge.textContent = 'VALID FORMAT';
      resultBadge.style.color = 'var(--accent-green)';
      resultBadge.style.background = 'var(--accent-green-dim)';
      resultBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      resultBadge.textContent = 'UNSTANDARDIZED';
      resultBadge.style.color = 'var(--accent-amber)';
      resultBadge.style.background = 'var(--accent-amber-dim)';
      resultBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    }

    patternText.textContent = res.pattern_name || '--';
    confProvProb.textContent = `${(res.confidence.province_classification * 100).toFixed(1)}%`;

    // Render Debug Drawer
    renderDebugDrawer(res);
  }

  // --- Render Debug Inspection Drawer ---
  function renderDebugDrawer(res) {
    if (!state.isDebug || !res.debug) {
      debugDrawer.classList.remove('active');
      return;
    }

    debugDrawer.classList.add('active');
    const d = res.debug;

    dbgPoly.src = d.poly_overlay || '';
    dbgDeskew.src = d.deskewed || d.raw_warp || '';
    dbgComp.src = d.comp_overlay || '';
    dbgOcr.src = d.char_enhanced || '';

    // Render Province Probabilities Bar Chart
    dbgProvBars.innerHTML = '';
    if (d.prov_top5 && d.prov_top5.length > 0) {
      d.prov_top5.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'prob-item';
        row.innerHTML = `
          <div class="prob-text">
            <span>${item.name}</span>
            <span>${item.prob}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${Math.min(100, Math.max(0, item.prob))}%;"></div>
          </div>
        `;
        dbgProvBars.appendChild(row);
      });
    } else {
      dbgProvBars.innerHTML = '<div class="empty-state">No distribution data</div>';
    }
  }

  // --- RTSP / Live Stream Controller ---
  function setupRTSPStream() {
    btnConnectStream.addEventListener('click', () => {
      if (state.isStreaming) {
        stopStream();
      } else {
        startStream();
      }
    });
  }

  function startStream() {
    const source = rtspInput.value.trim() || '0';
    console.log(`[RTSP Stream] Connecting to source: ${source}`);

    state.isStreaming = true;
    btnConnectStream.innerHTML = '<span>Disconnect</span>';
    btnConnectStream.classList.add('btn-danger');
    btnConnectStream.classList.remove('btn-primary');

    streamStatusPill.textContent = 'Live Streaming';
    streamStatusPill.style.color = 'var(--accent-green)';

    streamPlaceholder.style.display = 'none';
    streamViewer.style.display = 'block';

    // Set stream src to MJPEG endpoint with debug parameter
    const streamUrl = `/api/stream/mjpeg?source=${encodeURIComponent(source)}&debug=${state.isDebug ? 'true' : 'false'}`;
    streamViewer.src = streamUrl;

    // Start polling latest detection every 600ms to update pipeline breakdown cards
    if (state.streamPollTimer) clearInterval(state.streamPollTimer);
    state.streamPollTimer = setInterval(async () => {
      try {
        const resp = await fetch('/api/stream/latest');
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.detected) {
            renderPipelineResult(data);
          }
        }
      } catch (e) {
        // Stream polling error ignored
      }
    }, 600);
  }

  function stopStream() {
    console.log('[RTSP Stream] Disconnecting');
    state.isStreaming = false;

    btnConnectStream.innerHTML = '<span>Connect</span>';
    btnConnectStream.classList.remove('btn-danger');
    btnConnectStream.classList.add('btn-primary');

    streamStatusPill.textContent = 'Standby';
    streamStatusPill.style.color = 'var(--text-secondary)';

    streamViewer.src = '';
    streamViewer.style.display = 'none';
    streamPlaceholder.style.display = 'flex';

    if (state.streamPollTimer) {
      clearInterval(state.streamPollTimer);
      state.streamPollTimer = null;
    }
  }

  // Initialize on DOM load
  document.addEventListener('DOMContentLoaded', init);
})();
