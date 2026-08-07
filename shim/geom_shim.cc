// Minimal flat-C shim exposing the LArSoft channel map / geometry to python via ctypes.
// Built once with g++; all logic above this layer is python.
#include "fhiclcpp/ParameterSet.h"
#include "larcorealg/Geometry/GeometryCore.h"
#include "larcorealg/Geometry/WireReadoutGeom.h"
#include "larcorealg/Geometry/StandaloneGeometrySetup.h"
#include "dunecore/Geometry/GeoObjectSorterAPA.h"
#include "dunecore/Geometry/DuneApaWireReadoutGeom.h"
#include "dunecore/Geometry/WireReadoutSorterAPA.h"
#include "larcorealg/Geometry/OpDetGeo.h"
#include "cetlib/filepath_maker.h"
#include "lardataalg/DetectorInfo/DetectorPropertiesStandard.h"
#include "lardataalg/DetectorInfo/DetectorClocksStandard.h"
#include "lardataalg/DetectorInfo/LArPropertiesStandard.h"
#include <memory>
#include <string>
#include <cstring>

namespace {
  std::unique_ptr<geo::GeometryCore> gGeom;
  std::unique_ptr<geo::WireReadoutGeom> gWire;
  std::unique_ptr<detinfo::LArPropertiesStandard> gLArProp;
  std::unique_ptr<detinfo::DetectorClocksStandard> gClocks;
  std::unique_ptr<detinfo::DetectorPropertiesStandard> gDetProp;
  std::string gErr;
}

extern "C" {

int evd_geom_init(const char* fclfile) {
  try {
    cet::filepath_lookup_after1 policy("FHICL_FILE_PATH");
    auto pset  = fhicl::ParameterSet::make(std::string(fclfile), policy);
    auto gp    = pset.get<fhicl::ParameterSet>("Geometry");
    auto wp    = pset.get<fhicl::ParameterSet>("WireReadout");
    gGeom = lar::standalone::SetupGeometry<geo::GeoObjectSorterAPA>(gp);
    gWire = std::make_unique<geo::DuneApaWireReadoutGeom>(
              wp, gGeom.get(), std::make_unique<geo::WireReadoutSorterAPA>());

    // Detector properties are optional: geometry alone is enough to draw wires,
    // but without them we cannot convert TDC ticks to a drift coordinate.
    try {
      gLArProp = std::make_unique<detinfo::LArPropertiesStandard>(
                   pset.get<fhicl::ParameterSet>("LArProperties"));
      gClocks  = std::make_unique<detinfo::DetectorClocksStandard>(
                   pset.get<fhicl::ParameterSet>("DetectorClocks"));
      gDetProp = std::make_unique<detinfo::DetectorPropertiesStandard>(
                   pset.get<fhicl::ParameterSet>("DetectorProperties"),
                   gGeom.get(), gWire.get(), gLArProp.get(),
                   std::set<std::string>{"InheritNumberTimeSamples"});
    } catch (std::exception const& e) {
      gErr = std::string("detector properties unavailable: ") + e.what();
      gDetProp.reset();
    }
    return 0;
  } catch (std::exception const& e) { gErr = e.what(); return 1; }
    catch (...) { gErr = "unknown"; return 2; }
}

const char* evd_geom_error() { return gErr.c_str(); }
unsigned evd_n_channels() { return gWire ? gWire->Nchannels() : 0; }
unsigned evd_n_tpc()      { return gGeom ? gGeom->NTPC() : 0; }
unsigned evd_n_cryo()     { return gGeom ? gGeom->Ncryostats() : 0; }
const char* evd_det_name(){ static std::string s; s = gGeom ? gGeom->DetectorName() : ""; return s.c_str(); }

// Enumerate every wire: fills caller-allocated arrays. Returns number written.
// Arrays must hold at least evd_n_wires_total() entries.
long evd_n_wires_total() {
  if (!gGeom || !gWire) return -1;
  long n = 0;
  for (auto const& pid : gWire->Iterate<geo::PlaneID>()) n += gWire->Nwires(pid);
  return n;
}

long evd_dump_wires(unsigned* cryo, unsigned* tpc, unsigned* plane, unsigned* wire,
                    unsigned* chan, int* view, int* signal,
                    double* x0, double* y0, double* z0,
                    double* x1, double* y1, double* z1) {
  if (!gGeom || !gWire) return -1;
  long i = 0;
  for (auto const& pid : gWire->Iterate<geo::PlaneID>()) {
    auto const& pgeo = gWire->Plane(pid);
    unsigned nw = gWire->Nwires(pid);
    for (unsigned w = 0; w < nw; ++w) {
      geo::WireID id(pid, w);
      auto const& wg = gWire->Wire(id);
      auto s = wg.GetStart(); auto e = wg.GetEnd();
      cryo[i]=id.Cryostat; tpc[i]=id.TPC; plane[i]=id.Plane; wire[i]=id.Wire;
      chan[i]=gWire->PlaneWireToChannel(id);
      view[i]=(int)pgeo.View();
      signal[i]=(int)gWire->SignalType(id);
      x0[i]=s.X(); y0[i]=s.Y(); z0[i]=s.Z();
      x1[i]=e.X(); y1[i]=e.Y(); z1[i]=e.Z();
      ++i;
    }
  }
  return i;
}

// ---------------------------------------------------------------------------
// Detector properties: enough to map TDC ticks onto the drift coordinate.
//
// ConvertTicksToX is linear in tick for a fixed plane, so we export the
// intercept and slope per plane rather than a per-hit call.  That keeps the
// python side vectorised: x = x0[plane] + slope[plane] * tick.
// ---------------------------------------------------------------------------

int evd_has_detprop() { return gDetProp ? 1 : 0; }

long evd_n_planes() {
  if (!gWire) return -1;
  long n = 0;
  for (auto const& pid : gWire->Iterate<geo::PlaneID>()) { (void)pid; ++n; }
  return n;
}

// Per readout plane: identity, view, wire count, drift linearisation and the
// plane's own position/normal (useful for drawing and for drift direction).
long evd_dump_planes(unsigned* cryo, unsigned* tpc, unsigned* plane,
                     int* view, unsigned* nwires,
                     double* x0, double* slope,
                     double* px, double* py, double* pz,
                     double* nx, double* ny, double* nz) {
  if (!gGeom || !gWire) return -1;
  bool const haveDrift = gDetProp && gClocks;
  long i = 0;
  for (auto const& pid : gWire->Iterate<geo::PlaneID>()) {
    auto const& pg = gWire->Plane(pid);
    cryo[i] = pid.Cryostat; tpc[i] = pid.TPC; plane[i] = pid.Plane;
    view[i] = (int)pg.View();
    nwires[i] = gWire->Nwires(pid);
    if (haveDrift) {
      auto const dp = gDetProp->DataFor(gClocks->DataForJob());
      double a = dp.ConvertTicksToX(0.0, pid);
      double b = dp.ConvertTicksToX(1000.0, pid);
      x0[i] = a; slope[i] = (b - a) / 1000.0;
    } else { x0[i] = 0.0; slope[i] = 0.0; }
    auto c = pg.GetCenter(); auto nrm = pg.GetNormalDirection();
    px[i] = c.X(); py[i] = c.Y(); pz[i] = c.Z();
    nx[i] = nrm.X(); ny[i] = nrm.Y(); nz[i] = nrm.Z();
    ++i;
  }
  return i;
}

// Active-volume box of every TPC, for drawing the detector outline.
long evd_dump_tpcs(unsigned* cryo, unsigned* tpc,
                   double* xmin, double* ymin, double* zmin,
                   double* xmax, double* ymax, double* zmax) {
  if (!gGeom) return -1;
  long i = 0;
  for (auto const& tid : gGeom->Iterate<geo::TPCID>()) {
    auto const& t = gGeom->TPC(tid);
    auto const& box = t.ActiveBoundingBox();
    cryo[i] = tid.Cryostat; tpc[i] = tid.TPC;
    xmin[i] = box.MinX(); ymin[i] = box.MinY(); zmin[i] = box.MinZ();
    xmax[i] = box.MaxX(); ymax[i] = box.MaxY(); zmax[i] = box.MaxZ();
    ++i;
  }
  return i;
}

// ---------------------------------------------------------------------------
// Photon detectors. Optical hits carry only a channel number, so without a
// channel -> position map the optical view can plot nothing but an index.
// ---------------------------------------------------------------------------

long evd_n_opchannels() {
  if (!gWire) return -1;
  try { return gWire->NOpChannels(); } catch (...) { return 0; }
}

long evd_dump_opchannels(unsigned* chan, unsigned* opdet,
                         double* x, double* y, double* z) {
  if (!gWire) return -1;
  unsigned int n = 0;
  try { n = gWire->NOpChannels(); } catch (...) { return 0; }
  long i = 0;
  for (unsigned int c = 0; c < n; ++c) {
    try {
      auto const& od = gWire->OpDetGeoFromOpChannel(c);
      auto p = od.GetCenter();
      chan[i] = c;
      opdet[i] = gWire->OpDetFromOpChannel(c);
      x[i] = p.X(); y[i] = p.Y(); z[i] = p.Z();
      ++i;
    } catch (...) {
      // a channel with no detector behind it: skip rather than abort the dump
    }
  }
  return i;
}

long evd_n_tpcs_total() {
  if (!gGeom) return -1;
  long n = 0;
  for (auto const& tid : gGeom->Iterate<geo::TPCID>()) { (void)tid; ++n; }
  return n;
}

} // extern "C"
