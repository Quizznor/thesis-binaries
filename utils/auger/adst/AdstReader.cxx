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
  { 90000, 100 },
  { 90001, 175 },{ 90002, 175 },
  { 90003, 250 },{ 90004, 250 },{ 90005, 250 },
  { 90006, 325 },{ 90007, 325 },{ 90008, 325 },{ 90009, 325 },
  { 90010, 400 },{ 90011, 400 },{ 90012, 400 },{ 90013, 400 },{ 90014, 400 },
  { 90015, 475 },{ 90016, 475 },{ 90017, 475 },{ 90018, 475 },{ 90019, 475 },
  { 90020, 475 },
  { 90021, 550 },{ 90022, 550 },{ 90023, 550 },{ 90024, 550 },{ 90025, 550 },
  { 90026, 550 },{ 90027, 550 },
  { 90028, 625 },{ 90029, 625 },{ 90030, 625 },{ 90031, 625 },{ 90032, 625 },
  { 90033, 625 },{ 90034, 625 },{ 90035, 625 },
  { 90036, 700 },{ 90037, 700 },{ 90038, 700 },{ 90039, 700 },{ 90040, 700 },
  { 90041, 700 },{ 90042, 700 },{ 90043, 700 },{ 90044, 700 },
  { 90045, 775 },{ 90046, 775 },{ 90047, 775 },{ 90048, 775 },{ 90049, 775 },
  { 90050, 775 },{ 90051, 775 },{ 90052, 775 },{ 90053, 775 },{ 90054, 775 },
  { 90055, 850 },{ 90056, 850 },{ 90057, 850 },{ 90058, 850 },{ 90059, 850 },
  { 90060, 850 },{ 90061, 850 },{ 90062, 850 },{ 90063, 850 },{ 90064, 850 },
  { 90065, 850 },
  { 90066, 925 },{ 90067, 925 },{ 90068, 925 },{ 90069, 925 },{ 90070, 925 },
  { 90071, 925 },{ 90072, 925 },{ 90073, 925 },{ 90074, 925 },{ 90075, 925 },
  { 90076, 925 },{ 90077, 925 },
  { 90078, 1000 },{ 90079, 1000 },{ 90080, 1000 },{ 90081, 1000 },{ 90082, 1000 },
  { 90083, 1000 },{ 90084, 1000 },{ 90085, 1000 },{ 90086, 1000 },{ 90087, 1000 },
  { 90088, 1000 },{ 90089, 1000 },{ 90090, 1000 },
  { 90091, 1075 },{ 90092, 1075 },{ 90093, 1075 },{ 90094, 1075 },{ 90095, 1075 },
  { 90096, 1075 },{ 90097, 1075 },{ 90098, 1075 },{ 90099, 1075 },{ 90100, 1075 },
  { 90101, 1075 },{ 90102, 1075 },{ 90103, 1075 },{ 90104, 1075 },
  { 90105, 1150 },{ 90106, 1150 },{ 90107, 1150 },{ 90108, 1150 },{ 90109, 1150 },
  { 90110, 1150 },{ 90111, 1150 },{ 90112, 1150 },{ 90113, 1150 },{ 90114, 1150 },
  { 90115, 1150 },{ 90116, 1150 },{ 90117, 1150 },{ 90118, 1150 },{ 90119, 1150 },
  { 90120, 1225 },{ 90121, 1225 },{ 90122, 1225 },{ 90123, 1225 },{ 90124, 1225 },
  { 90125, 1225 },{ 90126, 1225 },{ 90127, 1225 },{ 90128, 1225 },{ 90129, 1225 },
  { 90130, 1225 },{ 90131, 1225 },{ 90132, 1225 },{ 90133, 1225 },{ 90134, 1225 },
  { 90135, 1225 },
  { 90136, 1300 },{ 90137, 1300 },{ 90138, 1300 },{ 90139, 1300 },{ 90140, 1300 },
  { 90141, 1300 },{ 90142, 1300 },{ 90143, 1300 },{ 90144, 1300 },{ 90145, 1300 },
  { 90146, 1300 },{ 90147, 1300 },{ 90148, 1300 },{ 90149, 1300 },{ 90150, 1300 },
  { 90151, 1300 },{ 90152, 1300 },
  { 90153, 1375 },{ 90154, 1375 },{ 90155, 1375 },{ 90156, 1375 },{ 90157, 1375 },
  { 90158, 1375 },{ 90159, 1375 },{ 90160, 1375 },{ 90161, 1375 },{ 90162, 1375 },
  { 90163, 1375 },{ 90164, 1375 },{ 90165, 1375 },{ 90166, 1375 },{ 90167, 1375 },
  { 90168, 1375 },{ 90169, 1375 },{ 90170, 1375 },
  { 90171, 1450 },{ 90172, 1450 },{ 90173, 1450 },{ 90174, 1450 },{ 90175, 1450 },
  { 90176, 1450 },{ 90177, 1450 },{ 90178, 1450 },{ 90179, 1450 },{ 90180, 1450 },
  { 90181, 1450 },{ 90182, 1450 },{ 90183, 1450 },{ 90184, 1450 },{ 90185, 1450 },
  { 90186, 1450 },{ 90187, 1450 },{ 90188, 1450 },{ 90189, 1450 }, 
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
    // DetectorGeometry detectorGeometry = DetectorGeometry();                       // contains SPDistance
    // recEventFile.ReadDetectorGeometry(detectorGeometry);

    // create csv file streams
    ofstream traceFile(csvTraceFile.string(), std::ios::out | std::ios::binary);

    // binaries of the generated shower
    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV

    traceFile.write(reinterpret_cast<const char*>(&showerEnergy), sizeof showerEnergy);
    traceFile.write(reinterpret_cast<const char*>(&showerZenith), sizeof showerZenith);

    // const auto showerAxis = genShower.GetAxisSiteCS();
    // const auto showerCore = genShower.GetCoreSiteCS();

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

  if (argc == 2 ) ExtractDataFromAdstFiles(fs::path(argv[1]));
  else if (argc > 2) {
    int start = std::atoi(argv[1]) + 3;
    int end = std::atoi(argv[2]) + 3;
  
    for (int i=start; i < end; i++)
    {
      std::cout << "Processing " << i - 2 << "/" << end - 2 << ": " << argv[i] << "\n";
      ExtractDataFromAdstFiles(fs::path(argv[i]));
    }
  }

  return 0;

}