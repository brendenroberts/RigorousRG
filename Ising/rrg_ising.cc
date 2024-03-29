#include "rrg.h"
#include "itensor/mps/sites/spinhalf.h"
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[]) {
    if(argc != 2) { std::cerr << "usage: " << argv[0] << " config_file" << std::endl; return 1; }
    std::ostringstream ss;
    std::map<string,string> configParams;
    std::ofstream logFile;
    std::cout.fill('0');
    
    std::ifstream configFile;
    configFile.open(argv[1]);
    if(configFile.is_open()) parseConfig(configFile,configParams);
    else { std::cerr << "error opening config file" << std::endl; return 1; }
    configFile.close();

    // RRG & AGSP parameters
    const size_t N   = stoul(configParams.at("N")); // system size
    const double tau = stod(configParams.at("tau"));  // AGSP temperature
    const size_t M   = stoul(configParams.at("M")); // num Trotter steps
    const size_t s   = stoul(configParams.at("s")); // formal s param
    const size_t D   = stoul(configParams.at("D")); // formal D param
    const double eps = stod(configParams.at("eps")); // MPVS error tolerance
    const bool doLanczos = true; // diag restricted Hamiltonian iteratively?
    
    // Hamitonian parameters
    const double J = stod(configParams.at("J")); // Ising interaction strength
    const double g = stod(configParams.at("g")); // transverse field strength
    const double h = stod(configParams.at("h")); // longitudinal field strength
    const auto hSpGen = [](size_t x) -> SiteSet { return SpinHalf(x,{"ConserveQNs",false}); };
    auto const& hs = hSpGen(N);

    // IO stream stuff for setting up output filenames
    auto configId = configParams.find("id");
    ss.setf(std::ios::fixed);
    ss.fill('0');
    if(configId == configParams.end())
        ss << argv[0] << "_N" << std::setw(3) << N;
    else ss << (*configId).second;
    auto dbFilename = ss.str();
    ss << "_J" << std::setprecision(2) << J << "_g" << std::setprecision(2) << g
       << "_h" << std::setprecision(2) << h;
    auto id = ss.str();
    if(configParams.find("log") != configParams.end())
        if(auto doLog = configParams.at("log") ; doLog == "true" || doLog == "True" || doLog == "1") {
            ss << ".log";
            auto logFilename = ss.str();
            std::cout << "writing output to " << logFilename << std::endl;
            logFile.open(logFilename);
            std::cout.rdbuf(logFile.rdbuf()); // redirect cout to log file buffer
            }
    std::ostringstream().swap(ss);

    // Create MPO for H using AutoMPO functionality
    AutoMPO autoH(hs);
    for(auto i = 0 ; static_cast<size_t>(i) < N ; ++i) {
        if(static_cast<size_t>(i) != N-1)
            autoH += -J*4.0,"Sz",i+1,"Sz",i+2;
        autoH += -g*2.0,"Sx",i+1;
        autoH += -h*2.0,"Sz",i+1;
        }
    auto H = toMPO(autoH,{"Exact",true});

    // Generate AGSP thermal operator exp(-H/t)
    auto K = Trotter(tau,M,autoH,1e-10);
    std::cout << "maximum AGSP bond dim = " << maxLinkDim(K) << std::endl;

    // Do RRG, obtaining a single MPVS object
    auto [res,time] = rrg(autoH,K,configParams.at("n"),hSpGen,{"Cutoff",eps,"ExtDim",s,"OpDim",D,"Iterative",doLanczos});

    // CLEANUP: extract MPS from MPVS and do some rounds of DMRG
    auto [extIndex,eSite] = findExt(res);
    auto [U,Dg] = diagHermitian(inner(res,res,eSite),{"Tags","Ext"});
    Dg.apply([](Real r) {return 1.0/sqrt(r);});
    res.ref(eSite) *= U*dag(Dg);
    res.ref(eSite).noPrime();
    auto [P,S] = diagHermitian(-inner(res,H,res,eSite),{"Tags","Ext"});
    res.ref(eSite) *= P;
    extIndex = findIndex(res(eSite),"Ext");

    for(auto j : range1(N-1)) std::cout << rightLinkIndex(res,j).dim() << " ";
    std::cout << std::endl;
    std::cout << "gs: " << std::fixed << std::setprecision(5) << -S.elt(1,1) << " gaps: ";
    for(auto j : range1(extIndex.dim()-1))
        std::cout << std::fixed << std::setprecision(4) << -S.elt(j+1,j+1)+S.elt(1,1) << " ";
    std::cout << std::endl;

    using ePair = pair<double,MPS>;
    vector<ePair> eigenstates;
    for(auto i : range1(dim(extIndex))) {
        auto fc = MPS(res);
        fc.ref(eSite) *= setElt(dag(extIndex)=i);
        fc.orthogonalize({"Cutoff",epx,"RespectDegenerate",true});
        fc.normalize();
        eigenstates.push_back({inner(fc,H,fc),fc});
        }

    std::cout << "DMRG steps:" << std::endl;
    auto nSweep = 2lu , nDMRG = 8lu;
    auto t3 = std::chrono::high_resolution_clock::now();
    for(auto i : range(nDMRG)) {
        dmrgMPO(H,eigenstates,nSweep,{"Exclude",true,"Cutoff",eps,"Penalty",5.0});
        std::sort(eigenstates.begin(), eigenstates.end(),[](auto const& a, auto const& b) { return a.first < b.first; });
        std::cout << "gs: " << std::fixed << std::setprecision(5) << eigenstates.at(0).first << " gaps: ";
        for(auto j : range1(eigenstates.size()-1))
            std::cout << std::fixed << std::setprecision(4) << eigenstates.at(j).first-eigenstates.at(0).first << " ";
        std::cout << std::endl;
        }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto tDMRG = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);
    std::cout << "dmrg elapsed: " << std::fixed << std::setprecision(0) << tDMRG.count() << " s" << std::endl;

    // EXIT: write out spectral data, save ground state in each sector to disk
    std::ostringstream().swap(ss);
    ss << id << "_sites.dat";
    auto sitesFilename = ss.str();
    writeToFile(sitesFilename,hs);
    std::ostringstream().swap(ss);

    std::ostringstream dbEntry;
    dbEntry.setf(std::ios::fixed);
    dbEntry.fill('0');
    dbEntry << "# N J g h s D tau M time E0 ..." << std::endl;
    dbEntry << std::setw(2) << N << " " << std::setprecision(2) << J << " " << std::setprecision(2) << g
            << " " << std::setprecision(2) << h << " " << std::setw(2) << s << " " << std::setw(2) << D
            << " " << std::setprecision(3) << tau << " " << std::setw(4) << M << " " << std::setprecision(2) << time << " ";
    for(auto j : range(s)) {
        dbEntry << std::setprecision(16) << eigenstates.at(j).first << " ";

        std::ostringstream().swap(ss);
        ss.fill('0');
        ss << id << "_e" << std::setw(2) << j << ".dat";
        auto stateFilename = ss.str();
        writeToFile(stateFilename,eigenstates.at(j).second);
        }
    dbEntry << std::endl;

    // Minimize amount of time spent with db file open (but don't bother with locking)
    std::ofstream dbFile(dbFilename,std::fstream::app);
    dbFile << dbEntry.str();
    dbFile.flush();
    dbFile.close();

    return 0;
    }
