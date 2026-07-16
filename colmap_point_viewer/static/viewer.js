import * as THREE from "https://esm.sh/three@0.180.0";
import { OrbitControls } from "https://esm.sh/three@0.180.0/examples/jsm/controls/OrbitControls.js";

function makeMatrix4(values) {
  const matrix = new THREE.Matrix4();
  matrix.set(
    values[0][0], values[0][1], values[0][2], values[0][3],
    values[1][0], values[1][1], values[1][2], values[1][3],
    values[2][0], values[2][1], values[2][2], values[2][3],
    values[3][0], values[3][1], values[3][2], values[3][3],
  );
  return matrix;
}

function colmapCameraToThreePose(c2w) {
  const basisChange = new THREE.Matrix4().makeScale(1, -1, -1);
  return c2w.clone().multiply(basisChange);
}


function quantileSorted(values, q) {
  if (values.length === 0) return 0;
  const index = Math.min(values.length - 1, Math.max(0, Math.round(q * (values.length - 1))));
  return values[index];
}

function computeRobustPointBox(positions, pointCount, keptFraction = 0.9) {
  const trim = Math.max(0, Math.min(0.5, (1 - keptFraction) / 2));
  const axes = [[], [], []];
  for (let i = 0; i < pointCount; i += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = positions[i * 3 + axis];
      if (Number.isFinite(value)) axes[axis].push(value);
    }
  }

  const mins = [];
  const maxs = [];
  for (const values of axes) {
    if (values.length === 0) return null;
    values.sort((a, b) => a - b);
    mins.push(quantileSorted(values, trim));
    maxs.push(quantileSorted(values, 1 - trim));
  }

  return new THREE.Box3(
    new THREE.Vector3(mins[0], mins[1], mins[2]),
    new THREE.Vector3(maxs[0], maxs[1], maxs[2]),
  );
}

function buildFrustumGeometry(c2w, H, W, fx, fy, cx, cy, scale) {
  const center = new THREE.Vector3().setFromMatrixPosition(c2w);
  const rotation = new THREE.Matrix3().setFromMatrix4(c2w);
  const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) => {
    const dir = new THREE.Vector3((u - cx) / fx, (v - cy) / fy, 1).normalize();
    dir.applyMatrix3(rotation).normalize();
    return center.clone().addScaledVector(dir, scale);
  });
  const points = [];
  for (const corner of corners) points.push(center, corner);
  for (let i = 0; i < 4; i += 1) points.push(corners[i], corners[(i + 1) % 4]);
  return new THREE.BufferGeometry().setFromPoints(points);
}

function updateStatus(statusEl, pointCount, cameraCount, selectedCamera) {
  statusEl.textContent = `points: ${pointCount} | cameras: ${cameraCount} | selected camera: ${selectedCamera ?? "none"}`;
}

function stringifyLogValue(value) {
  if (value instanceof Error) return `${value.name}: ${value.message}\n${value.stack ?? ""}`;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function isBenignResizeObserverMessage(values) {
  return values.some((value) => (typeof value === "string" && value.includes("ResizeObserver loop completed with undelivered notifications"))
    || (value && typeof value.message === "string" && value.message.includes("ResizeObserver loop completed with undelivered notifications")));
}

function createDebugConsole(el) {
  const panel = el.querySelector(".colmap-point-viewer__debug-panel");
  const logEl = el.querySelector(".colmap-point-viewer__debug-log");
  const toggleButton = el.querySelector(".colmap-point-viewer__debug-toggle");
  const clearButton = el.querySelector(".colmap-point-viewer__debug-clear");

  const append = (level, ...values) => {
    if (level === "error" && isBenignResizeObserverMessage(values)) {
      level = "warn";
    }
    const timestamp = new Date().toLocaleTimeString();
    const line = `[${timestamp}] ${level}: ${values.map(stringifyLogValue).join(" ")}`;
    logEl.textContent += `${line}\n`;
    logEl.scrollTop = logEl.scrollHeight;
    if (level === "error") {
      panel.hidden = false;
    }
  };

  toggleButton.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
  });
  clearButton.addEventListener("click", () => {
    logEl.textContent = "";
  });

  return { append, panel, logEl, toggleButton, clearButton };
}

export function render({ model, el }) {
  el.innerHTML = model.get("html_template");
  el.style.display = "block";
  el.style.width = "100%";
  el.style.maxWidth = "none";
  el.style.margin = "0";
  el.style.background = model.get("background");
  const styledAncestors = [];
  let ancestor = el.parentElement;
  for (let i = 0; i < 3 && ancestor; i += 1) {
    styledAncestors.push({
      element: ancestor,
      background: ancestor.style.background,
      padding: ancestor.style.padding,
      margin: ancestor.style.margin,
      width: ancestor.style.width,
      maxWidth: ancestor.style.maxWidth,
    });
    ancestor.style.background = model.get("background");
    ancestor.style.padding = "0";
    ancestor.style.margin = "0";
    ancestor.style.width = "100%";
    ancestor.style.maxWidth = "none";
    ancestor = ancestor.parentElement;
  }

  const root = el.querySelector(".colmap-point-viewer");
  const statusEl = el.querySelector(".colmap-point-viewer__status");
  const prevCameraButton = el.querySelector(".colmap-point-viewer__camera-prev");
  const nextCameraButton = el.querySelector(".colmap-point-viewer__camera-next");
  const backToOrbitButton = el.querySelector(".colmap-point-viewer__back-to-orbit");
  const canvasContainer = el.querySelector(".colmap-point-viewer__canvas-container");
  const debugConsole = createDebugConsole(el);

  root.style.width = "100%";
  root.style.maxWidth = "none";
  root.style.height = model.get("height");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.background = model.get("background");
  root.style.borderRadius = "6px";
  root.style.overflow = "hidden";
  root.style.border = "1px solid rgba(0, 0, 0, 0.45)";
  statusEl.style.color = "#e8edf2";
  statusEl.style.font = "12px/1.4 system-ui, sans-serif";
  statusEl.style.flex = "1 1 auto";
  statusEl.parentElement.style.padding = "6px 10px";
  statusEl.parentElement.style.background = "rgba(0, 0, 0, 0.22)";
  statusEl.parentElement.style.display = "flex";
  statusEl.parentElement.style.alignItems = "center";
  statusEl.parentElement.style.gap = "10px";
  prevCameraButton.style.cursor = "pointer";
  nextCameraButton.style.cursor = "pointer";
  backToOrbitButton.style.cursor = "pointer";
  debugConsole.toggleButton.style.cursor = "pointer";
  debugConsole.panel.style.maxHeight = "220px";
  debugConsole.panel.style.overflow = "auto";
  debugConsole.panel.style.background = "#101318";
  debugConsole.panel.style.borderTop = "1px solid rgba(255, 255, 255, 0.12)";
  debugConsole.panel.style.borderBottom = "1px solid rgba(255, 255, 255, 0.12)";
  debugConsole.panel.style.padding = "6px 10px";
  debugConsole.logEl.style.margin = "6px 0 0";
  debugConsole.logEl.style.whiteSpace = "pre-wrap";
  debugConsole.logEl.style.color = "#f2f5f7";
  debugConsole.logEl.style.font = "12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace";
  canvasContainer.style.flex = "1 1 auto";
  canvasContainer.style.minHeight = "0";
  canvasContainer.style.position = "relative";

  const previousConsoleError = console.error;
  const previousConsoleWarn = console.warn;
  console.error = (...args) => {
    debugConsole.append("error", ...args);
    previousConsoleError.apply(console, args);
  };
  console.warn = (...args) => {
    debugConsole.append("warn", ...args);
    previousConsoleWarn.apply(console, args);
  };
  const onWindowError = (event) => debugConsole.append("error", event.message, event.filename, event.lineno, event.colno, event.error);
  const onUnhandledRejection = (event) => debugConsole.append("error", "Unhandled promise rejection", event.reason);
  window.addEventListener("error", onWindowError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);

  const pc = model.get("pc");
  const colors = model.get("pc_color");
  const c2ws = model.get("c2ws");
  const pointCount = pc.length;
  const cameraCount = c2ws.length;
  updateStatus(statusEl, pointCount, cameraCount, null);
  debugConsole.append("info", `viewer init: points=${pointCount}, cameras=${cameraCount}`);
  debugConsole.append("info", `intrinsics: H=${model.get("H")}, W=${model.get("W")}, fx=${model.get("fx")}, fy=${model.get("fy")}, cx=${model.get("cx")}, cy=${model.get("cy")}`);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(model.get("background"));

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  const gl = renderer.getContext();
  debugConsole.append("info", `WebGL renderer: ${gl.getParameter(gl.RENDERER)}`);
  debugConsole.append("info", `WebGL vendor: ${gl.getParameter(gl.VENDOR)}`);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  canvasContainer.appendChild(renderer.domElement);

  const mainCamera = new THREE.PerspectiveCamera(60, 1, 0.001, 100000);
  mainCamera.up.set(0, -1, 0);
  const controls = new OrbitControls(mainCamera, renderer.domElement);
  controls.enableDamping = true;

  const positions = new Float32Array(pointCount * 3);
  const colorValues = new Float32Array(pointCount * 3);
  for (let i = 0; i < pointCount; i += 1) {
    positions.set(pc[i], i * 3);
    colorValues.set(colors[i], i * 3);
  }

  const pointGeometry = new THREE.BufferGeometry();
  pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  pointGeometry.setAttribute("color", new THREE.BufferAttribute(colorValues, 3));
  pointGeometry.computeBoundingBox();
  const pointMaterial = new THREE.PointsMaterial({ vertexColors: true, size: model.get("point_size") });
  const points = new THREE.Points(pointGeometry, pointMaterial);
  scene.add(points);

  const fullBbox = pointGeometry.boundingBox ?? new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  const bbox = computeRobustPointBox(positions, pointCount, 0.9) ?? fullBbox;
  const center = bbox.getCenter(new THREE.Vector3());
  const size = bbox.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 1e-3);
  debugConsole.append("info", `full bbox min=${fullBbox.min.toArray().join(", ")} max=${fullBbox.max.toArray().join(", ")}`);
  debugConsole.append("info", `robust bbox 90% min=${bbox.min.toArray().join(", ")} max=${bbox.max.toArray().join(", ")} radius=${radius}`);
  debugConsole.append("info", `robust bbox center=${center.toArray().join(", ")}`);
  const bboxHelper = new THREE.Box3Helper(bbox, 0xffffff);
  scene.add(bboxHelper);
  controls.target.copy(center);
  mainCamera.position.copy(center).add(new THREE.Vector3(radius, -radius, radius));
  mainCamera.lookAt(center);
  controls.update();
  mainCamera.near = Math.max(radius / 10000, 0.001);
  mainCamera.far = Math.max(radius * 100, 1000);
  mainCamera.updateProjectionMatrix();

  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const frustumScale = radius * 0.08;
  const markerRadius = radius * 0.012;
  const markerGeometry = new THREE.SphereGeometry(markerRadius, 12, 8);
  const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xffcc00 });
  const frustumMaterial = new THREE.LineBasicMaterial({ color: 0x66ccff });
  const markers = [];
  const frustums = [];
  const cameraPoses = [];
  const H = model.get("H");
  const W = model.get("W");
  const fx = model.get("fx");
  const fy = model.get("fy");
  const cx = model.get("cx");
  const cy = model.get("cy");

  c2ws.forEach((values, i) => {
    const c2w = makeMatrix4(values);
    cameraPoses.push(c2w);
    const frustum = new THREE.LineSegments(buildFrustumGeometry(c2w, H, W, fx, fy, cx, cy, frustumScale), frustumMaterial);
    frustums.push(frustum);
    scene.add(frustum);

    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    marker.position.setFromMatrixPosition(c2w);
    marker.userData.cameraIndex = i;
    markers.push(marker);
    scene.add(marker);
  });

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const fovY = 2 * Math.atan(H / (2 * fy)) * 180 / Math.PI;
  let selectedCameraIndex = null;
  let savedOrbitView = null;

  function saveOrbitView() {
    return {
      position: mainCamera.position.clone(),
      quaternion: mainCamera.quaternion.clone(),
      up: mainCamera.up.clone(),
      fov: mainCamera.fov,
      near: mainCamera.near,
      far: mainCamera.far,
      target: controls.target.clone(),
    };
  }

  function setCameraModeButtonsHidden(hidden) {
    prevCameraButton.hidden = hidden;
    nextCameraButton.hidden = hidden;
    backToOrbitButton.hidden = hidden;
  }

  function setCameraHelpersVisible(visible) {
    for (const frustum of frustums) frustum.visible = visible;
    for (const marker of markers) marker.visible = visible;
  }

  function restoreOrbitView() {
    if (savedOrbitView === null) return;
    selectedCameraIndex = null;
    setCameraModeButtonsHidden(true);
    bboxHelper.visible = true;
    setCameraHelpersVisible(true);
    controls.enabled = true;
    mainCamera.position.copy(savedOrbitView.position);
    mainCamera.quaternion.copy(savedOrbitView.quaternion);
    mainCamera.up.copy(savedOrbitView.up);
    mainCamera.fov = savedOrbitView.fov;
    mainCamera.near = savedOrbitView.near;
    mainCamera.far = savedOrbitView.far;
    controls.target.copy(savedOrbitView.target);
    mainCamera.updateProjectionMatrix();
    controls.update();
    updateStatus(statusEl, pointCount, cameraCount, null);
    debugConsole.append("info", "returned to orbit mode");
  }

  function selectCamera(cameraIndex, { saveOrbit = false } = {}) {
    if (cameraCount === 0) return;
    const wrappedIndex = (cameraIndex + cameraCount) % cameraCount;
    if (saveOrbit) savedOrbitView = saveOrbitView();
    selectedCameraIndex = wrappedIndex;
    controls.enabled = false;
    setCameraModeButtonsHidden(false);
    bboxHelper.visible = false;
    setCameraHelpersVisible(false);

    const c2w = cameraPoses[wrappedIndex];
    const threePose = colmapCameraToThreePose(c2w);
    mainCamera.matrixAutoUpdate = false;
    mainCamera.matrix.copy(threePose);
    mainCamera.matrix.decompose(mainCamera.position, mainCamera.quaternion, mainCamera.scale);
    mainCamera.matrixAutoUpdate = true;
    mainCamera.fov = fovY;
    mainCamera.aspect = W / H;
    mainCamera.near = Math.max(frustumScale / 100, 0.001);
    mainCamera.far = Math.max(radius * 100, 1000);
    mainCamera.updateProjectionMatrix();

    updateStatus(statusEl, pointCount, cameraCount, wrappedIndex);
    debugConsole.append("info", `selected camera mode: camera=${wrappedIndex}`);
  }

  function selectRelativeCamera(offset) {
    if (selectedCameraIndex === null) return;
    selectCamera(selectedCameraIndex + offset);
  }

  function onPointerDown(event) {
    if (selectedCameraIndex !== null) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, mainCamera);
    const hit = raycaster.intersectObjects(markers, false)[0];
    if (!hit) return;

    selectCamera(hit.object.userData.cameraIndex, { saveOrbit: true });
  }

  function onKeyDown(event) {
    if (selectedCameraIndex === null) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      selectRelativeCamera(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      selectRelativeCamera(1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      restoreOrbitView();
    }
  }

  let resizeFrame = null;
  function applyResize() {
    resizeFrame = null;
    const rect = canvasContainer.getBoundingClientRect();
    const width = Math.max(Math.floor(rect.width), 1);
    const height = Math.max(Math.floor(rect.height), 1);
    renderer.setSize(width, height, false);
    mainCamera.aspect = selectedCameraIndex === null ? width / height : W / H;
    mainCamera.updateProjectionMatrix();
    debugConsole.append("info", `resize: css=${width}x${height}, drawingBuffer=${renderer.domElement.width}x${renderer.domElement.height}, dpr=${renderer.getPixelRatio()}`);
  }

  function scheduleResize() {
    if (resizeFrame !== null) return;
    resizeFrame = requestAnimationFrame(applyResize);
  }

  const onPrevCameraClick = () => selectRelativeCamera(-1);
  const onNextCameraClick = () => selectRelativeCamera(1);
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  prevCameraButton.addEventListener("click", onPrevCameraClick);
  nextCameraButton.addEventListener("click", onNextCameraClick);
  backToOrbitButton.addEventListener("click", restoreOrbitView);
  window.addEventListener("keydown", onKeyDown);
  const resizeObserver = new ResizeObserver(scheduleResize);
  resizeObserver.observe(canvasContainer);
  scheduleResize();

  let animationFrame = null;
  function animate() {
    if (selectedCameraIndex === null) controls.update();
    renderer.render(scene, mainCamera);
    animationFrame = requestAnimationFrame(animate);
  }
  animate();

  return () => {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    if (resizeFrame !== null) cancelAnimationFrame(resizeFrame);
    renderer.domElement.removeEventListener("pointerdown", onPointerDown);
    prevCameraButton.removeEventListener("click", onPrevCameraClick);
    nextCameraButton.removeEventListener("click", onNextCameraClick);
    backToOrbitButton.removeEventListener("click", restoreOrbitView);
    window.removeEventListener("keydown", onKeyDown);
    for (const { element, background, padding, margin, width, maxWidth } of styledAncestors) {
      element.style.background = background;
      element.style.padding = padding;
      element.style.margin = margin;
      element.style.width = width;
      element.style.maxWidth = maxWidth;
    }
    resizeObserver.disconnect();
    window.removeEventListener("error", onWindowError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
    console.error = previousConsoleError;
    console.warn = previousConsoleWarn;
    controls.dispose();
    pointGeometry.dispose();
    pointMaterial.dispose();
    markerGeometry.dispose();
    markerMaterial.dispose();
    frustumMaterial.dispose();
    scene.traverse((object) => {
      if (object.geometry && object.geometry !== pointGeometry && object.geometry !== markerGeometry) object.geometry.dispose();
      if (object.material && ![pointMaterial, markerMaterial, frustumMaterial].includes(object.material)) object.material.dispose();
    });
    renderer.dispose();
    renderer.domElement.remove();
  };
}
