// Pauls stuff
#include <algorithm>
#include <iomanip>
#include <unordered_map>

// stl
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <functional>
#include <set>
#include <exception>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>

// from offline
#include <RecEventFile.h>
#include <DetectorGeometry.h>
#include <RecEvent.h>

#include <SdRecShower.h>
#include <SdRecStation.h>

#include <GenShower.h>
#include <Traces.h>
#include <TraceType.h>

#include <utl/Point.h>
#include <utl/UTMPoint.h>
#include <utl/ReferenceEllipsoid.h>
#include <utl/PhysicalConstants.h>
#include <utl/AugerUnits.h>
#include <utl/AugerCoordinateSystem.h>
#include <utl/CoordinateSystem.h>
#include <utl/CoordinateSystemPtr.h>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

using namespace utl;
namespace fs = boost::filesystem;

std::unordered_map<int, Double_t> showerPlaneMap = {
  { 90000, 200 },
  { 90001, 350 }, { 90002, 350 },
  { 90003, 500 }, { 90004, 500 }, { 90005, 500 },
  { 90006, 650 }, { 90007, 650 }, { 90008, 650 }, { 90009, 650 },
  { 90010, 800 }, { 90011, 800 }, { 90012, 800 }, { 90013, 800 }, { 90014, 800 },
  { 90015, 950 }, { 90016, 950 }, { 90017, 950 }, { 90018, 950 }, { 90019, 950 }, 
  { 90020, 950 },
  { 90021, 1100 }, { 90022, 1100 }, { 90023, 1100 }, { 90024, 1100 }, { 90025, 1100 }, 
  { 90026, 1100 }, { 90027, 1100 },
  { 90028, 1250 }, { 90029, 1250 }, { 90030, 1250 }, { 90031, 1250 }, { 90032, 1250 }, 
  { 90033, 1250 }, { 90034, 1250 }, { 90035, 1250 },
  { 90036, 1400 }, { 90037, 1400 }, { 90038, 1400 }, { 90039, 1400 }, { 90040, 1400 }, 
  { 90041, 1400 }, { 90042, 1400 }, { 90043, 1400 }, { 90044, 1400 },
};

void ExtractDataFromAdstFiles(fs::path pathToAdst)
{
  const auto csvTraceFile = pathToAdst.parent_path() / pathToAdst.filename().replace_extension("dat");

  // (2) start main loop
  RecEventFile     recEventFile(pathToAdst.string());
  RecEvent*        recEvent = nullptr;

  // will be assigned by root
  recEventFile.SetBuffers(&recEvent);

  for (unsigned int i = 0; i < recEventFile.GetNEvents(); ++i) 
  {
    // skip if event reconstruction failed
    if (recEventFile.ReadEvent(i) != RecEventFile::eSuccess) continue;

    // allocate memory for data
    const SDEvent& sdEvent = recEvent->GetSDEvent();                              // contains the traces
    const GenShower& genShower = recEvent->GetGenShower();                        // contains the shower
    DetectorGeometry detectorGeometry = DetectorGeometry();                       // contains SPDistance
    recEventFile.ReadDetectorGeometry(detectorGeometry);

    // create csv file streams
    ofstream traceFile(csvTraceFile.string(), std::ios::out | std::ios::binary);

    // binaries of the generated shower
    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV

    traceFile.write(reinterpret_cast<const char*>(&showerEnergy), sizeof showerEnergy);
    traceFile.write(reinterpret_cast<const char*>(&showerZenith), sizeof showerZenith);

    const auto showerAxis = genShower.GetAxisSiteCS();
    const auto showerCore = genShower.GetCoreSiteCS();  

    Detector detector = Detector();

    // loop over all triggered stations
    for (const auto& recStation : sdEvent.GetStationVector())
    {
      if (!recStation.IsDense()) continue;

      const auto stationId = recStation.GetId();
      // const auto SPD = detectorGeometry.GetStationAxisDistance(stationId, showerAxis, showerCore);  // in m
      const Double_t SPD = showerPlaneMap[stationId];
      traceFile.write(reinterpret_cast<const char*>(&stationId), sizeof stationId);
      traceFile.write(reinterpret_cast<const char*>(&SPD), sizeof SPD);

      const auto& traces = recStation.GetPMTTraces();
      for (const auto& trace : traces)
      {
        if (trace.GetType() != eTotalTrace) continue;
        if (trace.GetPMTId() == 4) continue;

        const auto peak = trace.GetPeak();
        const auto pmtid = trace.GetPMTId();
        const auto base = trace.GetBaseline();

        traceFile.write(reinterpret_cast<const char*>(&pmtid), sizeof pmtid);
        traceFile.write(reinterpret_cast<const char*>(&peak), sizeof peak);
        traceFile.write(reinterpret_cast<const char*>(&base), sizeof base);

        // const auto& vemTrace = trace.GetVEMComponent();
        const auto& vemTrace = trace.GetHighGainComponent();
        if (!vemTrace.size()) {
          const UShort_t dummy = 0;
          for (int i=0; i<2048; i++) {
            traceFile.write(reinterpret_cast<const char*>(&dummy), sizeof dummy);
          }
        } else {
          for (const auto& bin : vemTrace)
          {
            traceFile.write(reinterpret_cast<const char*>(&bin), sizeof bin);
          }
        }
      }
    }
    traceFile.close();
  }
}

int main(int argc, char** argv) 
{
  int start = 4170;
  int end = argc;
  for (int i=start; i < end; i++)
  {
    std::cout << "Processing " << i << "/" << argc-1 << ": " << argv[i] << "\n";
    ExtractDataFromAdstFiles(fs::path(argv[i]));
  }
  return 0;

}