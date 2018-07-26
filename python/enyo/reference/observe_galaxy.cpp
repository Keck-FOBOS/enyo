/***************************************************************************************************
    Author: Kyle Westfall
    Original Implementation: 12 Oct 2011

        Observe the toy galaxy model.

        The input/output surface brightness does not take into account any
        k-correction, but it does consider surface brightness dimming.

        From Bessell et al. (2008), flux zero-points as a function of band for Vega magnitudes:

            U       B       V       R       I       J       H       K
leff    0.366   0.438   0.545   0.641   0.798   1.220   1.630   2.190
Fl      417.5   632.0   363.1   217.7   112.6   31.47   11.38   3.961
ZPl    -0.152  -0.602   0.000   0.555   1.271   2.655   3.760   4.906

        leff is in microns
        Fl is in erg/cm^2/s/angstrom

        Extinction per airmass from the WEAVE ETC

            U      B      V      R      I      z
leff    0.366  0.438  0.545  0.641  0.798  0.950
ext      0.55   0.25   0.15   0.09   0.06   0.05 

        The fractional extinction is then pow(10, -0.4*ext*airmass) where airmass >= 1.0.
 
    Edits::
        - Redo the parameters so that the spatial and spectral PSFs are                 24 Oct 2011
        calculated directly (v1.1); not yet implemented!!
 
    To Do::
 
    Bugs::
 
***************************************************************************************************/

#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <cmath>
#include <string>
#include "vec.h"
#include "myfuncs.h"
#include "param.h"
#include "diskgalaxy.h"
#include "distance.h"
#include "fitsfuncs.h"
#include "mywcs.h"
#include "beam.h"
#include "interp_1d.h"
using namespace std;

//Globals

//Prototypes
bool initialize_galaxy(const ParSet &par, DiskGalaxy &gal, Vec_O_DP &rc_r);
double arcsec_per_pc(const ParSet &par);
double oblateness(double hR);
bool output_images(string oroot, const ParSet &par, DiskGalaxy &gal, int nx, int ny, int nd,
                   double pscale);
void output_fiber_data(string oroot, const ParSet &par, DiskGalaxy &gal, int nd, Mat_I_DP &fpos,
                       int nfs);
//double gaussint(double p, double prec, double &sig);
//double gausseval(double x);
double extinction(double cwl);
double fluxzp(double cwl);

int main (int argc, char *argv[]) {

        //Alert user that there are command line arguments available
        if (argc == 1)
            note("Command line options available (-h option prints a listing).");

        //Copy character string arguments to strings
        Vec_STR arglist(argc);          //Vector to hold command line arguments
        for (int i = 0; i < argc; ++i)
            arglist[i] = argv[i];

        //Command line variables
        int ntags = 9;          //Number of tags to search for
        Vec_STR tags(ntags);    //List of tags
        Vec_INT tagn(1, ntags); //Number of arguments for each tag
        tags[0] = "-I";         //ifile:        Input parameter file
        tags[1] = "-O";         //oroot:        Root name for output images
        tags[2] = "-p";         //pscale:       Pixel scale of the output image ("/pix)
        tags[3] = "-ni";        //nx, ny:       Pixel size of the image
        tagn[3] = 2;
        tags[4] = "-nd";        //nd:           Number of samples along the LOS for integration
        tags[5] = "-F";         //ffile:        File with fiber positions
        tags[6] = "-d";         //fibd:         Fiber diameter
        tags[7] = "-nf";        //nfs:          Number of samples per fiber diameter
        tags[ntags-1] = "-h";   //printhelp:    Print the command line options
        tagn[ntags-1] = 0;

        Mat_STR value;          //String representation of tag value
        readcommandline(arglist, argc, tags, tagn, value);

        //Set either command line values or ..                                  set defaults
        string ifile = value[0][0].size() > 0 ? value[0][1] :                  "observe_galaxy.par";
        string oroot = value[1][0].size() > 0 ? value[1][1] :                   "";
        double pscale = value[2][0].size() > 0 ? atof(value[2][1].c_str()) :    0.3;
        int nx = value[3][0].size() > 0 ? atoi(value[3][1].c_str()) :           512;
        int ny = value[3][0].size() > 0 ? atoi(value[3][2].c_str()) :           512;
        int nd = value[4][0].size() > 0 ? atoi(value[4][1].c_str()) :           100;
        string ffile = value[5][0].size() > 0 ? value[5][1] :                   "";
        double fibd = value[6][0].size() > 0 ? atof(value[6][1].c_str()) :      1.5;
        int nfs = value[7][0].size() > 0 ? atoi(value[7][1].c_str()) :          10;
        bool printhelp = value[ntags-1][0].size() != 0 ? true :                 false;

        //Provide help for command line arguments
        if (printhelp) {
            Vec_STR tagf(ntags);
            tagf[0] = tagf[1] = tagf[5] = "[file]";
            tagf[2] = tagf[4] = tagf[6] = tagf[7] = "[num]";
            tagf[3] = "[num] [num]";
            tagf[ntags-1] = "";

            Vec_STR tagd(ntags);
            tagd[0] = "Input parameter file (def: observe_galaxy.par)";
            tagd[1] = "Root name for output images";
            tagd[2] = "Pixel scale of the image (\"/pix; def: 0.3)";
            tagd[3] = "Pixel dimensions of the image (def: 512 512)";
            tagd[4] = "Number of samples along the LOS for integration (def: 100)";
            tagd[5] = "File with fiber positions for observations";
            tagd[6] = "Fiber diameter";
            tagd[7] = "Number of samples per fiber diameter for beam integration (def: 10)";
            tagd[ntags-1] = "Print this listing";

            commandlineoptions(tags, tagf, tagd);
            return 0;
        }

        //Check files
        if (!confirmifile(ifile, "Enter parameter file")) return 1;
        if (!confirmifile(ffile, "Enter file with fiber positions", false)) return 1;
//      confirmofile(ofits, "Enter output fits image", true);

        ParSet par(ifile);                              //Read the parameter file

        DiskGalaxy gal;
        Vec_DP rc_r;                                    //This needs to have the same scope as gal
        if (!initialize_galaxy(par, gal, rc_r)) return 1;

        if (ffile.size() == 0) {
            if (!output_images(oroot, par, gal, nx, ny, nd, pscale)) return 1;
            return 0;
        }

        //Read the fiber data
        Mat_DP fpos;
        int nf = 0;
        {
            Mat_DP tmp = readdat(ffile, true);
            if (tmp.ncols() != 3) {
                errormessage("Position file must have three columns: ID X Y");
                return 1;
            }
            nf = tmp.nrows();
            fpos.resize(nf, 4);
            for (int i = 0; i < nf; ++i) {
                fpos[i][0] = tmp[i][0];         //ID
                fpos[i][1] = tmp[i][1];         //X
                fpos[i][2] = tmp[i][2];         //Y
                fpos[i][3] = fibd;              //Fiber diameter
            }
        }

        //Produce the set of monochromatic observations
        output_fiber_data(oroot, par, gal, nd, fpos, nfs);

        return 0;
}

/* Initialize the DiskGalaxy object based on the data provided in the parameter
   list (see ../progfiles/observe_galaxy.par).  The list of rotation curve
   parameters is the only tricky part.  The list must be as follows:

   If the rotation curve type is TANH, PLAW, UNIV, PLEX, ISOT, or CVRT,
   rcp is just a comma separated list of the necessary parameters.

   This is also true for the NFWH parameters because the Hubble constant (h in
   units of 100 km/s/Mpc) and the distance are calculated based on other
   parameters in the input file.

   !! NOTE that the implementation of the NFW halo in RotCurve assumes that the
   !! object is at zero redshift.  This should be fixed for proper used of this
   !! rotation curve within this code.

   For the RLEG parameters, the first three listed parameters are the starting
   and ending radius and the polynomial order; the rest are then the
   coefficients of the polynomial.

   For the RING and LNRC parameters, the first N values are the bin radii, and
   the last N values are the velocities for each bin.  These are the only
   parameters necessary because the strict flag is never used (so the width of
   the last radius does not matter for the RING function as long as it is
   non-zero).  The number of rings is therefore half of the number of parameters
   provided.  An error will be returned if an even number of parameters is not
   given.
   
   Note that the units must be correct such that scale of the rotation curve
   matches the units of the input scale length (kpc).
*/
bool initialize_galaxy(const ParSet &par, DiskGalaxy &gal, Vec_O_DP &rc_r) {

        double msun, mu0, hR, hz, tau, hRd, hzd, truncr, truncz, incl, pa, sz0, hs, alpha;
        par.val("msun", msun);
        par.val("mu0", mu0);
        par.val("hR", hR);
        par.val("hz", hz);
        if (hz < 0) {
            double q;
            par.val("q", q);
            if (q < 0) q = oblateness(hR);
            hz = hR/q;
        }
        par.val("tau", tau);
        par.val("hRd", hRd);
        par.val("hzd", hzd);
        par.val("truncr", truncr);
        par.val("truncz", truncz);
        par.val("incl", incl);
        par.val("pa", pa);
        par.val("sz0", sz0);
        par.val("hs", hs);
        par.val("alpha", alpha);

        string rct;
        par.val("rctype", rct);
        int rctype = readmode_rc(rct);

        int nr = par.nelem("rcp");
        int order = 0;
        double rs = 0., re = 0.;
        double h = 0., D = 0.;
        Vec_DP rc_p;
        if (rctype == RLEG) {
            par.val("rcp", rs, 0);
            par.val("rcp", re, 1);
            par.val("rcp", order, 2);
            if (order != nr-3) {
                errormessage("Number of parameters does not match order.");
                return false;
            }
            rc_p.resize(nr-3);
            for (int i = 3; i < nr; ++i)
                par.val("rcp", rc_p[i-3], i);
        }
        else if (rctype == RING || rctype == LNRC) {
            if (nr % 2 > 0) {
                errormessage("Should provide even number of RING or LNRC parameters.");
                return false;
            }
            rc_r.resize(nr/2);
            rc_p.resize(nr/2);
            int i;
            for (i = 0; i < nr/2; ++i)
                par.val("rcp", rc_r[i], i);
            for ( ; i < nr; ++i)
                par.val("rcp", rc_p[i-nr/2], i);
        }
        else if (rctype == NFWH) {
            double z, H0, Omega_M, Omega_L;
            par.val("z", z);
            par.val("H0", H0);
            h = H0/100;
            par.val("Omega_M", Omega_M);
            par.val("Omega_L", Omega_L);
            D = comoving_distance(z, hubble_distance(H0), Omega_M, Omega_L);
        }
        else {
            rc_p.resize(nr);
            for (int i = 0; i < nr; ++i)
                par.val("rcp", rc_p[i], i);
        }

        try {
            gal = DiskGalaxy(msun, mu0, hR, hz, tau, hRd, hzd, truncr, truncz, incl, pa, rctype,
                             rc_p, h, D, &rc_r, rs, re, order, sz0, hs, alpha);
        }
        catch (Error err) { err.exe(); return false; }

        return true;
}

/* Get the angular scale of one parsec at the redshift of the object
*/
double arcsec_per_pc(const ParSet &par) {

        //Get the angular diameter on the sky
        double z, H0, Omega_M, Omega_L, da;
        par.val("z", z);
        par.val("H0", H0);
        par.val("Omega_M", Omega_M);
        par.val("Omega_L", Omega_L);
        da = angular_diameter_distance(z, hubble_distance(H0), Omega_M, Omega_L);

        printf(" distance: %9.4f\n", da);

        //da is in Mpc; convert radians to arcseconds
        return 1e-6/da * 180*60*60/M_PI;
}

/* Pulled from mllib.cpp. hR must be in kpc
  oblateness defined as hR/hz
*/
double oblateness(double hR) {
        return pow(10.0, 0.367*log10(hR)+0.708);
}

bool output_images(string oroot, const ParSet &par, DiskGalaxy &gal, int nx, int ny, int nd,
                   double pscale) {

        double s, e;
        gal.integration_limits(s, e);                   //Set the LOS integration limits

        Mat_DP intensity(0.0, nx, ny);                  //Allocate space for the image
        Mat_DP circvel(0.0, nx, ny);                    //Allocate space for the image
        Mat_DP stelvel(0.0, nx, ny);                    //Allocate space for the image
        Mat_DP stelsig(0.0, nx, ny);                    //Allocate space for the image
        WCS wcs(DSS, nx/2, ny/2, 0.0, 0.0, 0.0, 0.0, -pscale, pscale);  //Create the rough WCS

        //Factors needed to convert the luminosity to flux
        double msun, z, zp;
        par.val("zp", zp);
        par.val("msun", msun);
        par.val("z", z);

        //Create the images
        double pc2arcsec = arcsec_per_pc(par);          //Get the conversion from pc to arcsec
        printf("arcsec/pc: %12.4e\n", pc2arcsec);
        double x, y;                                    //Sky x and y position
        for (int j, i = 0; i < nx; ++i)
            for (j = 0; j < ny; ++j) {
//              printf(" %3d %3d %9.4f %9.4f\n", i, j, wcs.x_wcs(i+1,j+1), wcs.y_wcs(i+1,j+1));
                x = wcs.x_wcs(i+1,j+1) / pc2arcsec / 1e3;       //Sky X position in kpc
                y = wcs.y_wcs(i+1,j+1) / pc2arcsec / 1e3;       //Sky Y position in kpc
                gal.set_sky_position(x,y);

                gal.set_property(DG_SB);                        //Output intensity
                intensity[i][j] = gal.los_integrate(s,e,nd);    //Output is in L_sun/pc^2
                gal.set_property(DG_CV);                //Output intensity-weighted circular speed
                circvel[i][j] = gal.los_integrate(s,e,nd);      //Output is in km/s
                gal.set_property(DG_SV);                //Output intensity-weighted stellar rotation
                stelvel[i][j] = gal.los_integrate(s,e,nd);      //Output is in km/s
                gal.set_property(DG_SD);                //Output intensity-weighted vel dispersion
                stelsig[i][j] = gal.los_integrate(s,e,nd);      //Output is in (km/s)^2 L_sun/pc^2

                //Convert surface brightness to flux within the pixel
                intensity[i][j] = -2.5*log10(intensity[i][j])+msun+21.57;
                intensity[i][j] = pow(10.0, -0.4*(intensity[i][j]-zp))*SQR(pscale/SQR(1+z));

//              printf(" i,j,x,y,f: %d %d %9.4f %9.4f %12.4e\n", i, j, x, y, image[i][j]);
            }

        //Write the images and headers
        string ofits = oroot+".i.fits";
        if (!write2Dimage(ofits, intensity)) return false;
        if (!wcs.write2fits(ofits)) return false;

        ofits = oroot+".cv.fits";
        if (!write2Dimage(ofits, circvel)) return false;
        if (!wcs.write2fits(ofits)) return false;

        ofits = oroot+".sv.fits";
        if (!write2Dimage(ofits, stelvel)) return false;
        if (!wcs.write2fits(ofits)) return false;

        ofits = oroot+".ss.fits";
        if (!write2Dimage(ofits, stelsig)) return false;
        if (!wcs.write2fits(ofits)) return false;

        return true;
}

void output_fiber_data(string oroot, const ParSet &par, DiskGalaxy &gal, int nd, Mat_I_DP &fpos,
                       int nfs) {

        //Are all the beams of the same diameter?
        double fibd = fpos[0][3];
        int nf = fpos.nrows();
        bool renew_beam = false;
        for (int i = 1; i < nf; ++i)
            if (fabs(fpos[i][3]-fibd) > EPSDP) {
                renew_beam = true;
                break;
            }

        //Set up the beam object
        double seeing;
        par.val("seeing", seeing);                      //Grab the seeing (must be in arcsec)
        Beam beam(nfs, fibd, seeing, 0.01, 1.0);        //Initialize the Beam
        int ngrid = beam.ngrid();                       //Number of beam grid points
        double dx = beam.deltx();                       //Width of the pixel in x and y
        double dy = beam.delty();

        double msun, z, musky;
        par.val("msun", msun);
        par.val("z", z);
        par.val("musky", musky);

        //Quantities needed for the S/N calculation
        double cwl, specpsf, spatpsf, spatap, resolution;
        double efficiency, texp, readnoise, qe, telap, airmass;
        par.val("cwl", cwl);
        par.val("specpsf", specpsf);
        par.val("spatpsf", spatpsf);
        par.val("spatap", spatap);
        par.val("resolution", resolution);
        par.val("efficiency", efficiency);
        par.val("texp", texp);
        par.val("readnoise", readnoise);
        par.val("qe", qe);
        par.val("telap", telap);
        par.val("airmass", airmass);

        //Get the percentage of flux within the extraction aperture
        double spatsig = 1;                             //Number of sigma in the spatial aperture
        double specperc = gaussint(0.5, 1e-3, spatsig); //spatsig is a dummy variable
        double spatperc = gaussint(spatap, 1e-3, spatsig);

        double ext = pow(10.0, -0.4*extinction(cwl)*airmass);   //Get the atmospheric extinction
        double fzp = fluxzp(cwl);                               //Get the flux zeropoint

        //Get the angstroms per resolution element and the readnoise error
        //      assuming the spectral line is Gaussian and integrated within 1 FWHM
//      double dlambda = cwl/resolution*6.0/sig2fwhm;
        double dlambda = cwl/resolution;
        double rnerror = sqrt(specpsf * 2.0*spatsig/sig2fwhm*spatpsf)*readnoise;

        //Get the energy per photon in erg
        double phote = planck_h * C * 1e13 / cwl;

        //Conversion from flux in 1e-12 erg/cm^2/s/angstrom to extracted e-
        double f2e = 1e-12*dlambda*texp*telap*1e4*qe*efficiency*spatperc*specperc/phote;
        printf(" %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n", dlambda, texp, telap, qe, efficiency, spatperc, phote, f2e);

        double s, e;                            //Starting and ending depth for LOS integration

        double pc2arcsec = arcsec_per_pc(par);          //Get the conversion from pc to arcsec
        printf("15*arcsec2pc: %12.4e\n", 15.0/pc2arcsec);

        //For each fiber
        //  - Update the fiber diameter if necessary
        //  - Sample the galaxy to calculate
        //      0 - Source flux at the top of the atmosphere (erg/cm^2/s/angstrom)
        //      1 - Source flux at the top of the telescope (erg/cm^2/s/angstrom)
        //      2 - Sky flux at the top of the telescope (erg/cm^2/s/angstrom)
        //      3 - Effective aperture of the fiber (arcsec^2)
        //      4 - Intensity-weighted circular speed
        //      5 - Intensity-weighted stellar rotation speed
        //      6 - Intensity-weighted velocity dispersion
        //   These quantities are at the detector, integrated over the
        //   monochromatic resolution element
        //      7 - Source flux (e-) 
        //      8 - Sky flux (e-)
        //      9 - S/N per resolution element

        Mat_DP fiber_obs(0.0, nf, 10);          //Matrix to hold data
        double f, x, y, v, d, w, n;             //Temporary variables used in the calculation

        for (int j, i = 0; i < nf; ++i) {

            if (renew_beam && fabs(fpos[i][3]-fibd) > EPSDP) {          //Update the beam
                beam = Beam(nfs, fibd, seeing, 0.01, 1.0);
                dx = beam.deltx();
                dy = beam.delty();
                ngrid = beam.ngrid();
            }

            n = 0.0;                            //Reset normalization
            for (j = 0; j < ngrid; ++j) {

                x = (fpos[i][1]+beam.gridx(j)) / pc2arcsec / 1e3;       //Sky X position in kpc
                y = (fpos[i][2]+beam.gridy(j)) / pc2arcsec / 1e3;       //Sky Y position in kpc
                gal.set_sky_position(x,y);

                gal.integration_limits(s,e);                    //Set the LOS integration limits

                w = beam.gridw(j);                              //Get the beam weight
                fiber_obs[i][3] += w*dx*dy;                     //Effective aperture diameter

                //Source flux
                gal.set_property(DG_SB);                        //Output intensity
                f = gal.los_integrate(s,e,nd);                  //Output is in L_sun/pc^2
                //Convert surface brightness to flux within the pixel
                f = -2.5*log10(f)+msun+21.57;                   //Convert to mag/arcsec^2
                f = pow(10.0, -0.4*(f+21.1+fzp))*dx*dy/SQR(SQR(1+z))*1e12;      //Convert to flux

                fiber_obs[i][0] += w*f;                         //Add to total flux

                fiber_obs[i][2] += w*pow(10.0, -0.4*(musky+21.1+fzp))*dx*dy*1e12;       //Sky flux

                //Circular speed
                gal.set_property(DG_CV);                //Output intensity-weighted circular speed
                v = gal.los_integrate(s,e,nd);          //Output is in km/s
                fiber_obs[i][4] += w*f*v;

//              printf(" %9.4f %9.4f %12.4e %12.4e %12.4e", fpos[i][1]+beam.gridx(j), fpos[i][1]+beam.gridy(j), w, f, v);

                //Stellar rotation speed
                gal.set_property(DG_SV);                //Output intensity-weighted stellar rotation
                v = gal.los_integrate(s,e,nd);          //Output is in km/s
        
                //Stellar velocity dispersion
                gal.set_property(DG_SD);                //Output intensity-weighted vel dispersion
                d = gal.los_integrate(s,e,nd);          //Output is in km/s

                fiber_obs[i][5] += w*f*d*v;             //Stellar rotation speed numerator
                fiber_obs[i][6] += w*f*d*(d*d+v*v);     //Stellar velocity dispersion numerator

//              printf(" %12.4e %12.4e\n", v, d);

                n += w*f*d;                             //Normalization for stellar kinematics
            }

            fiber_obs[i][1] = fiber_obs[i][0]*ext;
            fiber_obs[i][4] = fiber_obs[i][0] > 0 ? fiber_obs[i][4]/fiber_obs[i][0] : 0.0;
            fiber_obs[i][5] = n > 0 ? fiber_obs[i][5]/n : 0.0;
            fiber_obs[i][6] = n > 0 ? sqrt(fiber_obs[i][6]/n - SQR(fiber_obs[i][5])) : 0.0;

            //Calculate the S/N of the extracted, monochromatic flux per resolution element
            fiber_obs[i][7] = fiber_obs[i][1]*f2e;
            fiber_obs[i][8] = fiber_obs[i][2]*f2e;
            fiber_obs[i][9] = fiber_obs[i][7]/sqrt(fiber_obs[i][7]+fiber_obs[i][8]+SQR(rnerror));
        }

        //Calculate the S/N of a 3600s observation at 25 mag/arcsec^2 
        double aeff = fiber_obs[0][3];          //Effective aperture of first fiber
        double fo = pow(10.0, -0.4*(25.0 + 21.1 + fzp))*1e12;
        double fs = pow(10.0, -0.4*(musky + 21.1 + fzp))*1e12;

        double foo = fo * f2e*aeff/texp;
        double fso = fs * f2e*aeff/texp;
        double sn = foo*3600 / sqrt( (foo+fso)*3600 + SQR(rnerror));

        //Calculate the surface brightness of 3600s observation at S/N=3
        double af = SQR(3600.0);
        double bf = -3*3*3600.0;
        double cf = -3*3*(fso*3600+SQR(rnerror));

        foo = (-bf + sqrt(SQR(bf)-4*af*cf))/2.0/af;
        fo = foo*texp/f2e/aeff;
        double mu = -2.5*log10(fo/1e12)-21.1-fzp;

        //Calculate the observing time required to reach S/N=3 for a surface
        //brightness of 25 mag/arcsec^2
        fo = pow(10.0, -0.4*(25.0 + 21.1 + fzp))*1e12;
        foo = fo * f2e*aeff/texp;
        double at = SQR(foo);
        double bt = -3*3*(foo+fso);
        double ct = -3*3*SQR(rnerror);
        double t = (-bt+sqrt(SQR(bt) - 4*at*ct))/2.0/at;

        //Print the results
        string ofile = oroot+".obs.db";
        timestamp(ofile, "#", true);
        FILE *fptr = fopen(ofile.c_str(), "a");
        fprintf(fptr, "#--------------------------------------------------------------------------------\n");
        fprintf(fptr, "#                                                               Galaxy Parameters\n");
        fprintf(fptr, "#--------------------------------------------------------------------------------\n#\n");
        gal.write_pars(fptr);
        fprintf(fptr, "#\n#--------------------------------------------------------------------------------\n");
        fprintf(fptr, "#                                                            Observing Parameters\n");
        fprintf(fptr, "#--------------------------------------------------------------------------------\n#\n");
        fprintf(fptr, "#                                 Beam sampling in X: %8.4f\n", dx);
        fprintf(fptr, "#                                 Beam sampling in Y: %8.4f\n", dy);
        fprintf(fptr, "#                         Number of beam grid points: %8d\n", ngrid);
        fprintf(fptr, "#                Sigma reached by spatial extraction: %8.4f\n", spatsig);
        fprintf(fptr, "#                        Fraction of light extracted: %8.4f\n", spatperc);
        fprintf(fptr, "#                     Percentage through atmospheric: %12.4e\n", ext);
        fprintf(fptr, "#                                     Flux zeropoint: %8.4f\n", fzp);
        fprintf(fptr, "#                   Angstroms per resolution element: %12.4e\n", cwl/resolution);
        fprintf(fptr, "#                         Integrated readnoise error: %8.4f\n", rnerror);
        fprintf(fptr, "#                           Energy in erg per photon: %12.4e\n", phote);
        fprintf(fptr, "#  Extracted electrons per 1e-12 erg/cm^2/s/angstrom: %12.4e\n", f2e);
        fprintf(fptr, "#\n#  Surface brightness reached in texp=3600s at S/N=3: %8.4f\n", mu);
        fprintf(fptr, "#   S/N reached in texp=3600s for mu=25 mag/arcsec^2: %8.4f\n", sn);
        fprintf(fptr, "# Time required to reach S/N=3 at mu=25 mag/arcsec^2: %12.4e\n", t);
        fprintf(fptr, "#\n#--------------------------------------------------------------------------------\n");
        fprintf(fptr, "#                                                                  Output results\n");
        fprintf(fptr, "#--------------------------------------------------------------------------------\n#\n");
        fprintf(fptr, "# Quantities are:\n");
        fprintf(fptr, "#          1 - ID number\n");
        fprintf(fptr, "#          2 - X position relative to galaxy center (arcsec)\n");
        fprintf(fptr, "#          3 - Y position relative to galaxy center (arcsec)\n");
        fprintf(fptr, "#          4 - Fiber diameter (arcsec)\n");
        fprintf(fptr, "#          5 - Effective fiber aperture (arcsec^2)\n");
        fprintf(fptr, "#          6 - Galaxy flux within effective fiber aperture at the top of the atmosphere (1e-12 erg/cm^2/s/angstrom)\n");
        fprintf(fptr, "#          7 - Mean surface brightness within the aperture (mag/arcsec^2)\n");
        fprintf(fptr, "#          8 - Galaxy flux at the top of the telescope (1e-12 erg/cm^2/s/angstrom)\n");
        fprintf(fptr, "#          9 - Sky flux at the top of the telescope (1e-12 erg/cm^2/s/angstrom)\n");
        fprintf(fptr, "#         10 - Number of extracted electrons at the detector from the galaxy\n");
        fprintf(fptr, "#         11 - Number of extracted electrons at the detector from the sky\n");
        fprintf(fptr, "#         12 - Signal-to-noise of the extracted resolution element\n");
        fprintf(fptr, "#         13 - Intensity-weighted, line-of-sight circular velocity (km/s)\n");
        fprintf(fptr, "#         14 - Intensity-weighted, line-of-sight stellar velocity (km/s)\n");
        fprintf(fptr, "#         15 - Intensity-weighted, line-of-sight stellar velocity dispersion (km/s)\n");
        fprintf(fptr, "#         16 - Flag for source-limited (1), background-limited (2), or detector-limited (3) observation\n");
        fprintf(fptr, "#\n#%4s %9s %9s %9s %9s %12s %9s %12s %12s %12s %12s %9s %9s %9s %9s %2s\n",
                "ID", "X", "Y", "D", "A_eff", "Obj_Flux", "SB_eff", "Ext_Flux", "Sky_Flux", "Obj_e",
                "Sky_e", "S/N", "CV", "SV", "SS", "L");

        double sb;
        int limitation;
        double dn = SQR(rnerror);

        for (int i = 0; i < nf; ++i) {
            sb = fiber_obs[i][0] > 0 ? -2.5*log10(fiber_obs[i][0]/fiber_obs[i][3]/1e12)-21.1-fzp : 0.0;
            if (fiber_obs[i][8] > fiber_obs[i][7] && fiber_obs[i][8] > dn)
                limitation = 2;                         //Background limited
            else if (dn > fiber_obs[i][7] && dn > fiber_obs[i][8])
                limitation = 3;                         //Detector limited
            else
                limitation = 1;                         //Source limited

            fprintf(fptr, " %4.0f %9.4f %9.4f %9.4f %9.4f %12.4e %9.4f %12.4e %12.4e %12.4e %12.4e %9.4f %9.4f %9.4f %9.4f %2d\n",
                    fpos[i][0], fpos[i][1], fpos[i][2], fpos[i][3], fiber_obs[i][3],
                    fiber_obs[i][0], sb, fiber_obs[i][1], fiber_obs[i][2], fiber_obs[i][7],
                    fiber_obs[i][8], fiber_obs[i][9], fiber_obs[i][4], fiber_obs[i][5],
                    fiber_obs[i][6], limitation);
        }
        fclose(fptr);
}

/* Taken from ../throughput/gaussint.cpp
   Determines the integral of a Gaussian to provided percentage of the peak.

   Parameters are:
        double p    - Fraction of the peak
        double prec - Precision of the calculation
        double &sig - Number of sigma for the extraction aperture
double gaussint(double p, double prec, double &sig) {

        //Number of sigma at this percentage
        sig = sqrt(2 * log(1/p));

        //Determine the integral
        int steps = 10;                         //Start with 10 steps
        double sum = 0, sumsave = 0;
        int i;
        Vec_DP gauss(steps + 1);
        Vec_DP x(steps + 1);

        for ( ; ; ) {

            //Get the integral over steps
            for (i = 0; i <= steps; ++i) {
                x[i] = i * sig / steps;
                gauss[i] = gausseval(x[i]);
            }
            for (i = 0; i < steps; ++i)
                sum += (x[i + 1] - x[i]) * gauss[i];
            //Multiply by 2 to account for only performing integral under one side of the gaussian
            sum *= 2;

            //Check if the precision has been reached
            if (fabs(sum - sumsave) < prec)
                break;
            else
                sumsave = sum;

            //Prepare for the next iteration
            sum = 0;
            steps *= 2;
            gauss.resize(steps + 1);
            x.resize(steps + 1);
        }

        return sum;
}

//Evaluates normalized gaussian at x
double gausseval(double x) {
        return 1 / sqrt(2 * M_PI) * exp(-x*x / 2);
}
*/

//Return the extinction magnitudes per airmass at a given central wavelength
double extinction(double cwl) {
        Vec_DP wave(6), ext(6);
        wave[0] = 3660;  ext[0] = 0.55;
        wave[1] = 4380;  ext[1] = 0.25;
        wave[2] = 5450;  ext[2] = 0.15;
        wave[3] = 6410;  ext[3] = 0.09;
        wave[4] = 7980;  ext[4] = 0.06;
        wave[5] = 9500;  ext[5] = 0.05;
        if (cwl < wave[0]) return ext[0];       //Extrapolation
        if (cwl > wave[5]) return ext[5];       //Extrapolation
        Spline_interp spl(wave, ext);
        return spl.interp(cwl);                 //Interpolation
}

//Return the zero-point needed to convert magnitudes to flux in units of erg/cm^2/s/angstrom
double fluxzp(double cwl) {
        Vec_DP wave(8), zp(8);
        wave[0] =  3660; zp[0] = -0.152;
        wave[1] =  4380; zp[1] = -0.602;
        wave[2] =  5450; zp[2] =  0.000;
        wave[3] =  6410; zp[3] =  0.555;
        wave[4] =  7980; zp[4] =  1.271;
        wave[5] = 12200; zp[5] =  2.655;
        wave[6] = 16300; zp[6] =  3.760;
        wave[7] = 21900; zp[7] =  4.906;
        if (cwl < wave[0]) return zp[0];        //Extrapolation
        if (cwl > wave[7]) return zp[7];        //Extrapolation
        Spline_interp spl(wave, zp);
        return spl.interp(cwl);                 //Interpolation
}
















