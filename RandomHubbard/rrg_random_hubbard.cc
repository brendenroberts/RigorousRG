#include "rrg.h"
#include "itensor/mps/sites/fermion.h"
#include <fstream>
#include <iostream>
#include <iomanip>

double dist(int i , int j , vector<double> dx , vector<double> dy) {
    return sqrt((i+dx.at(i)-j-dx.at(j))*(i+dx.at(i)-j-dx.at(j)) + (dy.at(i)-dy.at(j))*(dy.at(i)-dy.at(j)));
    }

int parseDisorder(string const& disFileName , vector<double> &dis , const int lineNum , const char delim = '\t') {
    dis.clear();
    std::ifstream disFile;
    disFile.open(disFileName);
    if(disFile.is_open()) {
        string line;
        for(auto i = 0 ; i <= lineNum ; ++i) std::getline(disFile,line);
        std::stringstream data(line);
        for(string val ; std::getline(data,val,delim) ; )
            dis.push_back(stod(val));
    } else { std::cerr << "error opening disorder file " << disFileName << std::endl; return 1; }
    disFile.close();
    
    return 0;
    }

int main(int argc, char *argv[]) {
    if(argc != 3) { std::cerr << "usage: " << argv[0] << " config_file line_num" << std::endl; return 1; }
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

    // User can request more than s states at the final step
    const size_t sts = configParams.find("states") == configParams.end() ? s : stoul(configParams.at("states"));

    // State targeting parameters (for continuous symmetry)
    const vector<int> targetQNs = {stoi(configParams.at("Nf"))}; 
    const int qnSpread = stoi(configParams.at("qnSpread")); // maximum allowed deviation for local blocks

    // Hamitonian parameters
    const double t = stod(configParams.at("t")); // fermion hopping
    const double lambda_ne = stod(configParams.at("lne"));
    const double lambda_ee = stod(configParams.at("lee"));
    const double Z = stod(configParams.at("Z"));
    const double zeta = stod(configParams.at("zeta"));
    const double f_ex = stod(configParams.at("fex"));
    const double a = stod(configParams.at("a")); // exp hopping correction
    const auto hSpGen = [](size_t x) -> SiteSet { return Fermion(x,{"ConserveQNs",true}); };
    auto const& hs = hSpGen(N);

    // Load disorder realization from text file
    const auto lineNum = atoi(argv[2]);
    vector<double> dx , dy;
    if(lineNum < 0) { dx = vector<double>(N,0.0); dy = vector<double>(N,0.0); }
    else {
        if(parseDisorder(configParams.at("dxFile"),dx,lineNum) != 0) return 1;
        if(parseDisorder(configParams.at("dyFile"),dy,lineNum) != 0) return 1;
        }
    auto r = [&dx,&dy](int i , int j) { return dist(i,j,dx,dy); };

    // IO stream stuff for setting up output filenames
    auto configId = configParams.find("id");
    ss.setf(std::ios::fixed);
    ss.fill('0');
    if(configId == configParams.end())
        ss << argv[0] << "_N" << std::setw(3) << N;
    else ss << (*configId).second;
    auto dbFilename = ss.str();
    auto id = ss.str();
    if(configParams.find("log") != configParams.end())
        if(auto doLog = configParams.at("log") ; doLog == "true" || doLog == "True" || doLog == "1") {
            ss << "_l" << std::setw(3) << lineNum << ".log";
            auto logFilename = ss.str();
            std::cout << "writing output to " << logFilename << std::endl;
            logFile.open(logFilename);
            std::cout.rdbuf(logFile.rdbuf()); // redirect cout to log file buffer
            }
    std::ostringstream().swap(ss);

    // Create MPO for H using AutoMPO functionality
    AutoMPO autoH(hs);
    for(auto i = 0 ; static_cast<size_t>(i) < N ; ++i) {
        if(static_cast<size_t>(i) < N-1) { 
            auto hop = exp(-(r(i,i+1)-1)/a);
            autoH += -t*hop,"C",i+1,"Cdag",i+2;
            autoH += -t*hop,"C",i+2,"Cdag",i+1;
            }
        
        auto pTerm = -1.0/zeta;
        for(auto j = 0 ; static_cast<size_t>(j) < N ; ++j)
            pTerm += 1.0/(r(i,j)+zeta);
        pTerm *= -lambda_ne*Z;
        autoH += pTerm,"N",i+1;

        for(auto j = 0 ; j < i ; ++j)
            autoH += lambda_ee*(abs(i-j) == 1 ? 1.0-f_ex : 1.0)/(r(i,j)+zeta),"N",i+1,"N",j+1;
        }
    auto H = toMPO(autoH,{"Cutoff",epx});

    // Generate AGSP thermal operator exp(-H/t)
    auto K = Trotter(tau,M,autoH,1e-10);
    std::cout << "maximum AGSP bond dim = " << maxLinkDim(K) << std::endl;

    // Do RRG, obtaining a single MPVS object
    auto [res,time] = rrg(autoH,K,configParams.at("n"),hSpGen,targetQNs,{"Cutoff",eps,"ExtDim",s,"OpDim",D,"ExtDimLast",sts,
                            "TruncateQNs",true,"QNSpread",qnSpread,"Iterative",doLanczos});

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
    for(auto n : range1(dim(extIndex))) std::cout << std::fixed << std::setprecision(8) << -S.elt(n,n) << " ";
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
    auto nSweep = 2lu , nDMRG = 6lu;
    auto t3 = std::chrono::high_resolution_clock::now();
    for(auto i : range(nDMRG)) {
        dmrgMPO(H,eigenstates,nSweep,{"Exclude",true,"Cutoff",eps});
        std::sort(eigenstates.begin(), eigenstates.end(),[](auto const& a, auto const& b) { return a.first < b.first; });
        for(auto const& v : eigenstates) std::cout << v.first << " ";
        std::cout << std::endl;
        }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto tDMRG = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);
    std::cout << "dmrg elapsed: " << std::fixed << std::setprecision(0) << tDMRG.count() << " s" << std::endl;
    if(logFile.is_open()) logFile.close();

    // EXIT: write out spectrum, save states to disk
    std::ostringstream().swap(ss);
    ss << id << "_sites.dat";
    auto sitesFilename = ss.str();
    writeToFile(sitesFilename,hs);
    std::ostringstream().swap(ss);

    std::ostringstream dbEntry;
    dbEntry.setf(std::ios::fixed);
    dbEntry.fill('0');
    for(auto i : range(eigenstates)) {
        dbEntry << "# N s D tau M t lne lee zeta fex a Nf time lineNum eIndex E" << std::endl;
        dbEntry << std::setw(3) << N << " " << std::setw(2) << s << " " << std::setw(2) << D
                << " " << std::setprecision(3) << tau << " " << std::setw(4) << M << " " << std::setw(4) << t
                << " " << std::setw(4) << lambda_ne << " " << std::setw(4) << lambda_ee << " " << std::setw(4) << zeta
                << " " << std::setw(4) << f_ex << " " << std::setw(4) << a << " " << std::setw(3) << -qn(extIndex,1).val(1)
                << " " << std::setprecision(2) << time << " " << std::setw(4) << lineNum << " " << std::setw(4) << i
                << " " << std::setprecision(16) << eigenstates.at(i).first << std::endl;

        std::ostringstream().swap(ss);
        ss.fill('0');
        ss << id << "_l" << std::setw(3) << lineNum << "_e" << std::setw(2) << i << ".dat";
        auto stateFilename = ss.str();
        writeToFile(stateFilename,eigenstates.at(i).second);
        }

    // Minimize amount of time spent with db file open (but don't bother with locking)
    std::ofstream dbFile(dbFilename,std::fstream::app);
    dbFile << dbEntry.str();
    dbFile.flush();
    dbFile.close();

    return 0;
    }
