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

/* GEN 1 MAP */
/* /cr/data01/filip/plots/gen1_dense.png */
/*
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
*/

/* GEN 2 MAP */
/* /cr/data01/filip/plots/gen2_dense.png */
std::unordered_map<int, Double_t> showerPlaneMap = {
{ 90200, 100 },{ 90201, 100 },{ 90202, 100 },{ 90203, 100 },
{ 90204, 170 },{ 90205, 170 },{ 90206, 170 },{ 90207, 170 },
{ 90208, 240 },{ 90209, 240 },{ 90210, 240 },{ 90211, 240 },{ 90212, 240 },
{ 90213, 310 },{ 90214, 310 },{ 90215, 310 },{ 90216, 310 },{ 90217, 310 },
{ 90218, 380 },{ 90219, 380 },{ 90220, 380 },{ 90221, 380 },{ 90222, 380 },{ 90223, 380 },
{ 90224, 450 },{ 90225, 450 },{ 90226, 450 },{ 90227, 450 },{ 90228, 450 },{ 90229, 450 },
{ 90230, 520 },{ 90231, 520 },{ 90232, 520 },{ 90233, 520 },{ 90234, 520 },{ 90235, 520 },{ 90236, 520 },
{ 90237, 590 },{ 90238, 590 },{ 90239, 590 },{ 90240, 590 },{ 90241, 590 },{ 90242, 590 },{ 90243, 590 },
{ 90244, 660 },{ 90245, 660 },{ 90246, 660 },{ 90247, 660 },{ 90248, 660 },{ 90249, 660 },{ 90250, 660 },{ 90251, 660 },
{ 90252, 730 },{ 90253, 730 },{ 90254, 730 },{ 90255, 730 },{ 90256, 730 },{ 90257, 730 },{ 90258, 730 },{ 90259, 730 },
{ 90260, 800 },{ 90261, 800 },{ 90262, 800 },{ 90263, 800 },{ 90264, 800 },{ 90265, 800 },{ 90266, 800 },{ 90267, 800 },{ 90268, 800 },
{ 90269, 870 },{ 90270, 870 },{ 90271, 870 },{ 90272, 870 },{ 90273, 870 },{ 90274, 870 },{ 90275, 870 },{ 90276, 870 },{ 90277, 870 },
{ 90278, 940 },{ 90279, 940 },{ 90280, 940 },{ 90281, 940 },{ 90282, 940 },{ 90283, 940 },{ 90284, 940 },{ 90285, 940 },{ 90286, 940 },{ 90287, 940 },
{ 90288, 1010 },{ 90289, 1010 },{ 90290, 1010 },{ 90291, 1010 },{ 90292, 1010 },{ 90293, 1010 },{ 90294, 1010 },{ 90295, 1010 },{ 90296, 1010 },{ 90297, 1010 },
{ 90298, 1080 },{ 90299, 1080 },{ 90300, 1080 },{ 90301, 1080 },{ 90302, 1080 },{ 90303, 1080 },{ 90304, 1080 },{ 90305, 1080 },{ 90306, 1080 },{ 90307, 1080 },{ 90308, 1080 },
{ 90309, 1150 },{ 90310, 1150 },{ 90311, 1150 },{ 90312, 1150 },{ 90313, 1150 },{ 90314, 1150 },{ 90315, 1150 },{ 90316, 1150 },{ 90317, 1150 },{ 90318, 1150 },{ 90319, 1150 },
{ 90320, 1220 },{ 90321, 1220 },{ 90322, 1220 },{ 90323, 1220 },{ 90324, 1220 },{ 90325, 1220 },{ 90326, 1220 },{ 90327, 1220 },{ 90328, 1220 },{ 90329, 1220 },{ 90330, 1220 },{ 90331, 1220 },
{ 90332, 1290 },{ 90333, 1290 },{ 90334, 1290 },{ 90335, 1290 },{ 90336, 1290 },{ 90337, 1290 },{ 90338, 1290 },{ 90339, 1290 },{ 90340, 1290 },{ 90341, 1290 },{ 90342, 1290 },{ 90343, 1290 },
{ 90344, 1360 },{ 90345, 1360 },{ 90346, 1360 },{ 90347, 1360 },{ 90348, 1360 },{ 90349, 1360 },{ 90350, 1360 },{ 90351, 1360 },{ 90352, 1360 },{ 90353, 1360 },{ 90354, 1360 },{ 90355, 1360 },{ 90356, 1360 },
{ 90357, 1430 },{ 90358, 1430 },{ 90359, 1430 },{ 90360, 1430 },{ 90361, 1430 },{ 90362, 1430 },{ 90363, 1430 },{ 90364, 1430 },{ 90365, 1430 },{ 90366, 1430 },{ 90367, 1430 },{ 90368, 1430 },{ 90369, 1430 },
{ 90370, 1500 },{ 90371, 1500 },{ 90372, 1500 },{ 90373, 1500 },{ 90374, 1500 },{ 90375, 1500 },{ 90376, 1500 },{ 90377, 1500 },{ 90378, 1500 },{ 90379, 1500 },{ 90380, 1500 },{ 90381, 1500 },{ 90382, 1500 },{ 90383, 1500 },
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

    // create csv file streams
    ofstream traceFile(csvTraceFile.string(), std::ios::out | std::ios::binary);

    // allocate memory for data
    const SDEvent& sdEvent = recEvent->GetSDEvent();                              // contains the traces
    const GenShower& genShower = recEvent->GetGenShower();                        // contains the shower
    // DetectorGeometry detectorGeometry = DetectorGeometry();                       // contains SPDistance
    // recEventFile.ReadDetectorGeometry(detectorGeometry);
    // const auto showerAxis = genShower.GetAxisSiteCS();
    // const auto showerCore = genShower.GetCoreSiteCS();

    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV

    traceFile.write(reinterpret_cast<const char*>(&showerEnergy), sizeof showerEnergy);
    traceFile.write(reinterpret_cast<const char*>(&showerZenith), sizeof showerZenith);

    // loop over all triggered stations
    for (const auto& recStation : sdEvent.GetStationVector())
    {
      if (!recStation.IsDense()) continue;

      const auto stationId = recStation.GetId();

      std::cout << stationId << "\n";
      // const auto SPD = detectorGeometry.GetStationAxisDistance(stationId, showerAxis, showerCore);  // in m
      const Double_t SPD = showerPlaneMap[stationId];
      const bool isTOT = recStation.IsTOT();
      traceFile.write(reinterpret_cast<const char*>(&stationId), sizeof stationId);
      traceFile.write(reinterpret_cast<const char*>(&SPD), sizeof SPD);
      traceFile.write(reinterpret_cast<const char*>(&isTOT), sizeof isTOT);

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