// ============================================================================
// InstaRoad — Multimodal export (S2 + S1 + topo) for South African sites
// Fixed version. Key changes vs. the original script:
//
//   1. S2, S1 and topo are exported as SEPARATE GeoTIFFs. The original stacked
//      uint16 (S2, 0-10000) with int16 (S1, negative dB) into one file, which
//      forces GEE to upcast the whole thing and defeats the "smaller files" goal.
//
//   2. S1 temporal mean is now computed in LINEAR POWER, then converted back to
//      dB. Averaging dB directly (the original .mean()) is the geometric mean of
//      power -> biased low, and is NOT the multitemporal speckle estimator that
//      Lin/Quin rely on.
//
//   3. S1 is no longer truncated by toInt16(). Raw dB is ~ -30..+1, so toInt16()
//      threw away the fractional dB — exactly the 1-3 dB road/background contrast
//      you need. It is now stored as round(dB * 100) in int16 (~0.01 dB, still
//      small files). >>> Divide by 100 at load time. <<<
//
//   4. S2: Lin-style median composite, native 0-10000 reflectance (uint16).
//      Split by native GSD into two files — the 10 m bands (B2,B3,B4,B8) at
//      scale 10 and the 20 m bands (B5,B6,B7,B8A,B11,B12) at scale 20 — so the
//      20 m bands keep their native resolution and you control the upsampling
//      to 10 m downstream (M1/bicubic). Scale by /10000 at load if you want 0-1.
//
//   5. Topo kept as float (was int16) so slope/aspect bands aren't truncated.
//      NOTE: topo is not part of the RQ1 modality ablation (M0-M3) — set
//      EXPORT_TOPO=false if you don't intend to use it.
//
// THINGS TO VERIFY (they live in utils you can't see from here):
//   - genSentinel2Data should apply QA60/SCL cloud masking AND a <20% scene
//     cloud filter to match Lin et al. 2.2.1. If it doesn't, add it in the util.
//   - Set S1_INPUT_IS_DB below to match what genSentinel1Data actually returns.
//   - Confirm the S1 bands come out ordered/named VHA, VVA, VHD, VVD.
// ============================================================================

var UniversalTools    = require("users/Laura_Chow77/default:utils/UniversalTools");
var ImgDataGeneration = require("users/Laura_Chow77/default:utils/ImgDataGeneration");
var PreprocessProduct = require("users/Laura_Chow77/default:utils/PreprocessProduct");

// ---- Config ----------------------------------------------------------------
var S1_INPUT_IS_DB    = true;   // set false if genSentinel1Data returns linear power
var S1_DB_SCALE       = 100;    // stored = round(dB * 100); divide by 100 on load
var EXPORT_TOPO       = true;   // topo is NOT in the RQ1 ablation — keep or drop
var DRIVE_FOLDER_S2_10M = 'Sen12_S2_10m';
var DRIVE_FOLDER_S2_20M = 'Sen12_S2_20m';
var DRIVE_FOLDER_S1     = 'Sen12_S1';
var DRIVE_FOLDER_TOPO   = 'Sen12_topo';

// ---- SAR helpers: temporal mean in linear power ----------------------------
function dbToPower(img) {
  img = ee.Image(img);
  return ee.Image(10).pow(img.divide(10)).rename(img.bandNames());
}
function powerToDb(img) {
  return ee.Image(img).log10().multiply(10);
}
// Correct temporal mean for speckle reduction: mean in power, then back to dB.
function s1TemporalMeanDb(coll) {
  var powerColl = S1_INPUT_IS_DB ? coll.map(dbToPower) : coll;
  return powerToDb(powerColl.mean());
}
function s1ToStoredInt16(db) {
  return db.multiply(S1_DB_SCALE).round().toInt16();
}

// ---- Bounding-box helper (unchanged) ---------------------------------------
function createRectangularBBox(lon, lat, widthKm, heightKm) {
  widthKm = widthKm || 25;
  heightKm = heightKm || 25;

  var zone = Math.floor((lon + 180) / 6) + 1;
  var isNorth = lat >= 0;
  var epsg = 'EPSG:' + (isNorth ? 32600 + zone : 32700 + zone);
  var localProj = ee.Projection(epsg);

  var centerPtUtm = ee.Geometry.Point([lon, lat]).transform(localProj, 1);
  var coords = centerPtUtm.coordinates();
  var cx = ee.Number(coords.get(0));
  var cy = ee.Number(coords.get(1));

  var xOffset = (widthKm * 1000) / 2;
  var yOffset = (heightKm * 1000) / 2;

  return ee.Geometry.Rectangle(
    [cx.subtract(xOffset), cy.subtract(yOffset), cx.add(xOffset), cy.add(yOffset)],
    localProj, false);
}

// ---- Imagery generation ----------------------------------------------------
function generateMulimodalImagery(geometry_bbox, start_date, end_date) {
  var s2_coll = ImgDataGeneration.genSentinel2Data(
    geometry_bbox, [start_date, end_date],
    ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']);

  var projection = s2_coll.first().select("B2").projection();
  var crs = projection.crs();

  var s1_coll_list = ImgDataGeneration.genSentinel1Data(geometry_bbox, [start_date, end_date], crs);

  // S1: linear-power temporal mean -> dB -> int16 stored as dB * S1_DB_SCALE
  var s1_asc  = s1ToStoredInt16(s1TemporalMeanDb(s1_coll_list[0]));
  var s1_desc = s1ToStoredInt16(s1TemporalMeanDb(s1_coll_list[1]));
  var s1 = s1_asc.addBands(s1_desc);

  // S2: Lin et al. median composite, native 0-10000 reflectance (uint16).
  // Split by native GSD: 10 m bands exported at 10 m, 20 m bands at 20 m so they
  // keep native resolution (upsample to 10 m downstream where you pick the kernel).
  var s2 = s2_coll.median().toUint16();
  var s2_10m = s2.select(['B4', 'B3', 'B2', 'B8']);
  var s2_20m = s2.select(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']);

  // Topo: float so any slope/aspect bands survive
  var topo = ImgDataGeneration.genTopoData()
    .resample('bilinear').reproject({crs: projection, scale: 10}).toFloat();

  return {s1: s1, s2_10m: s2_10m, s2_20m: s2_20m, topo: topo, projection: projection};
}

// ---- Export ----------------------------------------------------------------
function exportImage(image, name, suffix, folder, region, crs, scale) {
  Export.image.toDrive({
    image: image,
    region: region,
    scale: scale,
    crs: crs,
    maxPixels: 1e13,
    description: name + suffix,
    folder: folder,
    formatOptions: {cloudOptimized: true}
  });
}

function processAndExportBiome(name, geometry) {
  var geometry_bbox = geometry.bounds(1);
  var imagery = generateMulimodalImagery(geometry_bbox, start_date, end_date);

  var crs = imagery.projection.crs().getInfo();
  var exportRegion = geometry_bbox.bounds(1, crs);

  exportImage(imagery.s2_10m, name, '_S2_10m', DRIVE_FOLDER_S2_10M, exportRegion, crs, 10);
  exportImage(imagery.s2_20m, name, '_S2_20m', DRIVE_FOLDER_S2_20M, exportRegion, crs, 20);
  exportImage(imagery.s1,     name, '_S1',     DRIVE_FOLDER_S1,     exportRegion, crs, 10);
  if (EXPORT_TOPO) {
    exportImage(imagery.topo, name, '_topo', DRIVE_FOLDER_TOPO, exportRegion, crs, 10);
  }
}

// ----------------------------------------------------------------------------
// User-defined variables
// ----------------------------------------------------------------------------
var start_date = '2024-01-01';
var end_date   = '2025-01-01';
var visTargetBiome = 'Thohoyandou';

// NB: these are point LOCATIONS, not biomes. Before claiming "9 biomes" in the
// methods, verify the set actually spans them and your density strata.
var biomeGeometries = {
  'CapeTown':     createRectangularBBox(18.54, -33.975),
  'Malmesburg':   createRectangularBBox(18.72, -33.46),
  'Stellenbosch': createRectangularBBox(18.90, -33.90),
  'Worcester':    createRectangularBBox(19.46, -33.65),
  'PrinceAlbert': createRectangularBBox(22.03, -33.23),
  'TouwsRiver':   createRectangularBBox(20.03, -33.34),
  'Johannesburg': createRectangularBBox(28.04, -26.20),
  'Pretoria':     createRectangularBBox(28.19, -25.75),
  'Kuboes':       createRectangularBBox(16.99, -28.45),
  'BeaufortWest': createRectangularBBox(22.58, -32.35),
  'Skukuza':      createRectangularBBox(31.59, -24.99),
  'CathkinPark':  createRectangularBBox(29.52, -29.00),
  'Mtubatuba':    createRectangularBBox(32.18, -28.42),
  'Durban':       createRectangularBBox(30.92, -29.85),
  'Upington':     createRectangularBBox(21.24, -28.45),
  'Makhanda':     createRectangularBBox(26.53, -33.31),
  'Mbombela':     createRectangularBBox(30.97, -25.47),
  'Ngcwanguba':   createRectangularBBox(29.02, -31.95),
  'Thohoyandou':  createRectangularBBox(30.47, -22.94),
  'Nongoma':      createRectangularBBox(31.65, -27.89)
};

// ---- Dry-run / preview -----------------------------------------------------
var targetGeom = biomeGeometries[visTargetBiome];
Map.centerObject(targetGeom);
Map.addLayer(targetGeom.bounds(1), {color: 'FF0000'}, visTargetBiome);
Map.setOptions('SATELLITE');

var targetImagery = generateMulimodalImagery(targetGeom.bounds(1), start_date, end_date);
print('--- TARGETED DRY RUN INFO ---');
print('Selected Layer:', visTargetBiome);
print('Respective CRS:', targetImagery.projection.crs());
print('S1 bands (stored as dB * ' + S1_DB_SCALE + ', int16):', targetImagery.s1.bandNames());
print('S2 10 m bands (uint16, scale 10):', targetImagery.s2_10m.bandNames());
print('S2 20 m bands (uint16, scale 20):', targetImagery.s2_20m.bandNames());

// ---- Execute: queue export tasks for every site ----------------------------
Object.keys(biomeGeometries).forEach(function(biomeName) {
  processAndExportBiome(biomeName, biomeGeometries[biomeName]);
});