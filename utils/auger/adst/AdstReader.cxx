// Pauls stuff
#include <algorithm>
#include <iomanip>

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

using namespace std;
using namespace utl;
namespace fs = boost::filesystem;

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
      const auto stationId = recStation.GetId();
      const auto SPD = detectorGeometry.GetStationAxisDistance(stationId, showerAxis, showerCore);  // in m
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
  int start = 1;
  for (int i=start; i < argc; i++)
  {
    std::cout << "Processing " << i << "/" << argc-1 << ": " << argv[i] << "\n";
    ExtractDataFromAdstFiles(fs::path(argv[i]));
  }
  return 0;

}