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

function createDebugConsole(el) {
  const panel = el.querySelector(".colmap-point-viewer__debug-panel");
  const logEl = el.querySelector(".colmap-point-viewer__debug-log");
  const toggleButton = el.querySelector(".colmap-point-viewer__debug-toggle");
  const clearButton = el.querySelector(".colmap-point-viewer__debug-clear");

  const append = (level, ...values) => {
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
  const root = el.querySelector(".colmap-point-viewer");
  const statusEl = el.querySelector(".colmap-point-viewer__status");
  const canvasContainer = el.querySelector(".colmap-point-viewer__canvas-container");
  const debugConsole = createDebugConsole(el);

  root.style.height = model.get("height");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.background = model.get("background");
  root.style.borderRadius = "6px";
  root.style.overflow = "hidden";
  root.style.border = "1px solid rgba(255, 255, 255, 0.12)";
  statusEl.style.color = "#e8edf2";
  statusEl.style.font = "12px/1.4 system-ui, sans-serif";
  statusEl.parentElement.style.padding = "6px 10px";
  statusEl.parentElement.style.background = "rgba(0, 0, 0, 0.22)";
  statusEl.parentElement.style.display = "flex";
  statusEl.parentElement.style.alignItems = "center";
  statusEl.parentElement.style.gap = "10px";
  debugConsole.toggleButton.style.marginLeft = "auto";
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
  canvasContainer.appendChild(renderer.domElement);

  const mainCamera = new THREE.PerspectiveCamera(60, 1, 0.001, 100000);
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

  const bbox = pointGeometry.boundingBox ?? new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  const center = bbox.getCenter(new THREE.Vector3());
  const size = bbox.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 1e-3);
  debugConsole.append("info", `bbox min=${bbox.min.toArray().join(", ")} max=${bbox.max.toArray().join(", ")} radius=${radius}`);
  const bboxHelper = new THREE.Box3Helper(bbox, 0xffffff);
  scene.add(bboxHelper);
  controls.target.copy(center);
  mainCamera.position.copy(center).add(new THREE.Vector3(radius, -radius, radius));
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
  function onPointerDown(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, mainCamera);
    const hit = raycaster.intersectObjects(markers, false)[0];
    if (!hit) return;

    const cameraIndex = hit.object.userData.cameraIndex;
    const c2w = cameraPoses[cameraIndex];
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

    const target = new THREE.Vector3(0, 0, 1).applyMatrix4(c2w);
    const origin = new THREE.Vector3().setFromMatrixPosition(c2w);
    controls.target.copy(origin).add(target.sub(origin).normalize().multiplyScalar(radius * 0.25));
    controls.update();
    updateStatus(statusEl, pointCount, cameraCount, cameraIndex);
  }

  function resize() {
    const width = Math.max(canvasContainer.clientWidth, 1);
    const height = Math.max(canvasContainer.clientHeight, 1);
    renderer.setSize(width, height, false);
    mainCamera.aspect = width / height;
    mainCamera.updateProjectionMatrix();
  }

  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(canvasContainer);
  resize();

  let animationFrame = null;
  function animate() {
    controls.update();
    renderer.render(scene, mainCamera);
    animationFrame = requestAnimationFrame(animate);
  }
  animate();

  return () => {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    renderer.domElement.removeEventListener("pointerdown", onPointerDown);
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
