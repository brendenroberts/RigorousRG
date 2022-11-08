#include "rrg.h"
#include "itensor/mps/sites/fermion.h"
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
    const double t   = stod(configParams.at("t"));  // AGSP temperature
    const size_t M   = stoul(configParams.at("M")); // num Trotter steps
    const size_t s   = stoul(configParams.at("s")); // formal s param
    const size_t D   = stoul(configParams.at("D")); // formal D param
    const double eps = stod(configParams.at("eps")); // MPVS error tolerance
    const bool doLanczos = true; // diag restricted Hamiltonian iteratively?

    // Hamiltonian parameters
    const double Gamma = stod(configParams.at("Gamma")); // disorder strength
    const auto hSpGen = [](size_t x) -> SiteSet { return Fermion(x,{"ConserveQNs",true,"ConserveNf",false}); };
    auto const& hs = hSpGen(N);

    // Generate Hamiltonian couplings using C++ random library
    std::random_device rd;
    const unsigned long seed = (configParams.find("seed") == configParams.end() ? rd() : stoul(configParams.at("seed")));
    std::linear_congruential_engine<unsigned long, 6364136223846793005ul, 1442695040888963407ul, 0ul> gen(seed);
    std::uniform_real_distribution<double> udist(0.0,1.0);
    std::cout << "seed is " << seed << std::endl;

    vector<double> J(2*(N-1));
    for(auto i = 0u ; i < N-1 ; ++i) {
        J.at(2*i+0) = pow(udist(gen),Gamma);
        J.at(2*i+1) = pow(udist(gen),Gamma);
        }

    // IO stream stuff for setting up output filenames
    auto configId = configParams.find("id");
    ss.setf(std::ios::fixed);
    ss.fill('0');
    if(configId == configParams.end())
        ss << argv[0] << "_N" << std::setw(3) << N << "_s" << std::setw(2) << s << "_D" << std::setw(2) << D;
    else ss << (*configId).second;
    auto dbFilename = ss.str();
    ss << "_seed" << std::setw(10) << seed;
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
    for(auto i = 0 ; static_cast<size_t>(i) < N-1 ; ++i) {
        autoH += (J.at(2*i+0)+J.at(2*i+1)),"Cdag",i+1,   "C",i+2;
        autoH += (J.at(2*i+0)+J.at(2*i+1)),"Cdag",i+2,   "C",i+1;
        autoH += (J.at(2*i+0)-J.at(2*i+1)),"Cdag",i+1,"Cdag",i+2;
        autoH += (J.at(2*i+0)-J.at(2*i+1)),   "C",i+2,   "C",i+1;
        }
    auto H = toMPO(autoH,{"Exact",true});

    // Generate AGSP thermal operator exp(-H/t)
    auto K = Trotter(t,M,autoH,1e-10);
    std::cout << "maximum AGSP bond dim = " << maxLinkDim(K) << std::endl;

    // Do RRG, obtaining a single MPVS object
    auto [res,time] = rrg(autoH,K,configParams.at("n"),hSpGen,{"Cutoff",eps,"ExtDim",s,"OpDim",D,"Iterative",doLanczos});

    // CLEANUP: extract MPS from MPVS and optimize using DMRG
    auto [extIndex,eSite] = findExt(res);
    auto [U,Dg] = diagHermitian(inner(res,res),{"Tags","Ext"});
    Dg.apply([](Real r) {return 1.0/sqrt(r);});
    res.ref(eSite) *= U*dag(Dg);
    res.ref(eSite).noPrime();
    auto [P,S] = diagHermitian(-inner(res,H,res),{"Tags","Ext"});
    res.ref(eSite) *= P;
    extIndex = findIndex(res(eSite),"Ext");
    
    for(auto j : range1(N-1)) std::cout << rightLinkIndex(res,j).dim() << " ";
    std::cout << std::endl;
 
    auto eGS = -S.elt(1,1);
    std::cout << std::fixed << std::setprecision(14) << eGS << " (q:" << 0 << " gap:"
              << std::scientific << std::setprecision(4) << -S.elt(2,2)-eGS << ") ";
    for(auto q : range(nblock(extIndex)))
        if(q != 0)
            std::cout << std::scientific << std::setprecision(6) << -S.elt(2*q+1,2*q+1)-eGS << " (q:" << q << " gap:"
                      << std::scientific << std::setprecision(4) << -S.elt(2*q+2,2*q+2)+S.elt(2*q+1,2*q+1) << ") ";
    std::cout << std::endl;

    using ePair = pair<double,MPS>;
    vector<vector<ePair> > eigenspace;
    for(int q : range(nblock(extIndex))) {
        eigenspace.push_back(vector<ePair>());
        for(auto i : range1(2)) {
            auto fc = MPS(res);
            fc.ref(eSite) *= setElt(dag(extIndex)(2*q+i));
            fc.orthogonalize({"Cutoff",epx,"RespectDegenerate",true});
            fc.normalize();
            eigenspace.back().push_back({inner(fc,H,fc),fc});
            }
        }

    // Run DMRG until all energies/gaps are converged, up to a max of nDMRG rounds
    auto count = 0lu , nSweep = 6lu , nDMRG = 2*N; 
    auto minGap = 0.0 , convGS = 1.0;
    std::cout << "DMRG steps:" << std::endl;
    vector<double> ePrev(eigenspace.size());
    do {
        for(auto q : args(eigenspace)) {
            auto& sector = eigenspace.at(q);
            ePrev.at(q) = sector.front().first;
            dmrgMPO(H,sector,nSweep,{"Cutoff",epx});
            std::sort(sector.begin(),sector.end(),[](auto const& a , auto const& b) {return a.first < b.first;});
            }
        ++count;
        convGS = std::inner_product(eigenspace.begin(),eigenspace.end(),ePrev.begin(),0.0,
                                    [](double const& a , double const& b) { return std::max(a,b); }, // std::max<double> overloaded
                                    [](vector<ePair> const& a , double const& b) { return abs(b - a.front().first); });

        eGS = eigenspace.at(0).front().first;
        std::cout << std::fixed << std::setprecision(14) << eGS << " (q:" << 0 << " gap:"
                  << std::scientific << std::setprecision(4) << eigenspace.at(0).at(1).first-eGS << " m:"
                  << maxLinkDim(eigenspace.at(0).front().second) << ") ";
        minGap = eigenspace.at(0).at(1).first-eGS;
        for(auto q : args(eigenspace))
            if(q != 0) {
                minGap = std::min(minGap,std::min(eigenspace.at(q).front().first-eGS,eigenspace.at(q).at(1).first-eigenspace.at(q).front().first));
                std::cout << std::scientific << std::setprecision(6) << eigenspace.at(q).front().first-eGS << " (q:" << q << " gap:"
                          << std::scientific << std::setprecision(4) << eigenspace.at(q).at(1).first-eigenspace.at(q).front().first << " m:"
                          << maxLinkDim(eigenspace.at(q).front().second) << ") ";
                }
        std::cout << std::endl;
    } while(convGS > std::max(epx*N/4,std::min(eps,minGap*1e-2)) && count < nDMRG);
    if(logFile.is_open()) logFile.close();

    // EXIT: write out spectral data, save sector ground states to disk
    std::ostringstream().swap(ss);
    ss << id << "_sites.dat";
    auto sitesFilename = ss.str();
    writeToFile(sitesFilename,hs);
    std::ostringstream().swap(ss);

    std::ostringstream dbEntry;
    dbEntry.setf(std::ios::fixed);
    dbEntry.fill('0');
    for(auto q : args(eigenspace)) {
        dbEntry << "# N G s D t M seed time conv q E0 ..." << std::endl;
        dbEntry << std::setw(2) << N << " " << std::setprecision(2) << Gamma << " " << std::setw(2) << s
                << " " << std::setw(2) << D << " " << std::setprecision(3) << t << " " << std::setw(4) << M
                << " " << std::setw(10) << seed << " " << std::setprecision(2) << time
                << " " << std::setw(1) << (count == nDMRG ? 0 : 1)
                << " " << std::setw(1) << q << " ";
        auto const& sector = eigenspace.at(q);
        for(auto j : range(2)) {
            dbEntry << std::setprecision(16) << sector.at(j).first << " ";

            std::ostringstream().swap(ss);
            ss << id << "_q" << q << "_e" << j << ".dat";
            auto stateFilename = ss.str();
            writeToFile(stateFilename,sector.at(j).second);
            }
        dbEntry << std::endl;
        }
    
    // Minimize amount of time spent with db file open (but don't bother with locking)
    std::ofstream dbFile(dbFilename,std::fstream::app);
    dbFile << dbEntry.str();
    dbFile.flush();
    dbFile.close();

    return 0;
    }
