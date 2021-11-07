import Upwind as UP
import Lax_W as LW
import Lax_W_Periodic as LWp
import funcs as fn

import matplotlib.pyplot as plt

def main():

    # scheme params
    dx = 0.25 # m

    Lx = 100 # m
    H = 100 # m

    t1 = 0.5
    t2 = 1.0
    t3 = 2.0

    # wave params
    U = 10 # m/s
    A = 3 # m
    lam1 = 12 # m
    lam2 = 6 # m
    x1 = 30 # m
    x2 = 60 # m

    res_exact_t1 = fn.wave(U, A, lam1, Lx, dx, x1, x2, t1)
    res_exact_t2 = fn.wave(U, A, lam1, Lx, dx, x1, x2, t2)
    res_exact_t3 = fn.wave(U, A, lam1, Lx, dx, x1, x2, t3)

    res_exact_t1_hf = fn.wave(U, A, lam2, Lx, dx, x1, x2, t1)
    res_exact_t2_hf = fn.wave(U, A, lam2, Lx, dx, x1, x2, t2)
    res_exact_t3_hf = fn.wave(U, A, lam2, Lx, dx, x1, x2, t3)

    # BCs
    n_0 = 0
    n_Lx = 0
    
    CFL_1 = 0.5
    result_UP_1 = UP.Upwind(x1, x2, dx, Lx, t3, CFL_1, n_0, n_Lx, U, A, lam1)
    result_LaxW_1 = LW.LAXW(x1, x2, dx, Lx, t3, CFL_1, n_0, n_Lx, U, A, lam1)
    result_UP_1_hf = UP.Upwind(x1, x2, dx, Lx, t3, CFL_1, n_0, n_Lx, U, A, lam2)
    dt1 = result_UP_1[2]

    CFL_2 = 0.9
    result_UP_2 = UP.Upwind(x1, x2, dx, Lx, t3, CFL_2, n_0, n_Lx, U, A, lam1)
    result_LaxW_2 = LW.LAXW(x1, x2, dx, Lx, t3, CFL_2, n_0, n_Lx, U, A, lam1)
    result_UP_2_hf = UP.Upwind(x1, x2, dx, Lx, t3, CFL_2, n_0, n_Lx, U, A, lam2)
    dt2 = result_UP_2[2]

    CFL_3 = 1.0
    result_UP_3 = UP.Upwind(x1, x2, dx, Lx, t3, CFL_3, n_0, n_Lx, U, A, lam1)
    result_LaxW_3 = LW.LAXW(x1, x2, dx, Lx, t3, CFL_3, n_0, n_Lx, U, A, lam1)
    dt3 = result_UP_3[2]

    CFL_4 = 2.0
    result_UP_4 = UP.Upwind(x1, x2, dx, Lx, t3, CFL_4, n_0, n_Lx, U, A, lam1)
    result_LaxW_4 = LW.LAXW(x1, x2, dx, Lx, t3, CFL_4, n_0, n_Lx, U, A, lam1)
    dt4 = result_UP_4[2]

    result_LaxW_periodic = LWp.LAXW(x1, x2, dx, Lx, 60, CFL_3, n_0, n_Lx, U, A, lam1)


    # Upwind plotting, lambda = 12m
    fig1, (ax1, ax2, ax3) = plt.subplots(3,1)
    lw = 1

    im1_d = ax1.plot(result_UP_4[0], H+result_UP_4[1][int(t1/dt4)-1], 'r-', label='CFL = 2.0', linewidth=lw, alpha=0.2)
    im1_e = ax1.plot(result_UP_1[0], H+result_UP_1[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im1_f = ax1.plot(result_UP_1[0], H+res_exact_t1, label='Exact Solution', linewidth=lw)
    im1_a = ax1.plot(result_UP_1[0], H+result_UP_1[1][int(t1/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im1_b = ax1.plot(result_UP_2[0], H+result_UP_2[1][int(t1/dt2)-1], label='CFL = 0.9', linewidth=lw)
    im1_c = ax1.plot(result_UP_3[0], H+result_UP_3[1][int(t1/dt3)-1], label='CFL = 1.0', linewidth=lw)

    im2_d = ax2.plot(result_UP_4[0], H+result_UP_4[1][int(t2/dt4)-1], 'r-', label='CFL = 2.0', linewidth=lw, alpha=0.2)   
    im2_e = ax2.plot(result_UP_1[0], H+result_UP_1[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im2_f = ax2.plot(result_UP_1[0], H+res_exact_t2, label='Exact Solution', linewidth=lw)
    im2_a = ax2.plot(result_UP_1[0], H+result_UP_1[1][int(t2/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im2_b = ax2.plot(result_UP_2[0], H+result_UP_2[1][int(t2/dt2)-1], label='CFL = 0.9', linewidth=lw)
    im2_c = ax2.plot(result_UP_3[0], H+result_UP_3[1][int(t2/dt3)-1], label='CFL = 1.0',linewidth=lw)

    im3_d = ax3.plot(result_UP_4[0], H+result_UP_4[1][int(t3/dt4)-1], 'r-', label='CFL = 2.0', linewidth=lw, alpha=0.2)
    im3_e = ax3.plot(result_UP_1[0], H+result_UP_1[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im3_f = ax3.plot(result_UP_1[0], H+res_exact_t3, label='Exact Solution', linewidth=lw)
    im3_a = ax3.plot(result_UP_1[0], H+result_UP_1[1][int(t3/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im3_b = ax3.plot(result_UP_2[0], H+result_UP_2[1][int(t3/dt2)-1], label='CFL = 0.9', linewidth=lw)
    im3_c = ax3.plot(result_UP_3[0], H+result_UP_3[1][int(t3/dt3)-1], label='CFL = 1.0', linewidth=lw)

    ax1.set_title('t = 0.5s')
    ax1.set_ylim([H-A, H+A])
    ax1.set_xlabel('Length (m)')
    ax1.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax1.grid()
    ax1.legend()

    ax2.set_title('t = 1.0s')
    ax2.set_ylim([H-A, H+A])
    ax2.set_xlabel('Length (m)')
    ax2.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax2.legend()
    ax2.grid()

    ax3.set_title('t = 2.0s')
    ax3.set_ylim([H-A, H+A])
    ax3.set_xlabel('Length (m)')
    ax3.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax3.legend()
    ax3.grid()

    fig1.suptitle('Upwind, ' + r'$\lambda$' + ' = 12m')

    # Lax-W plotting
    fig2, (ax4, ax5, ax6) = plt.subplots(3,1)
    lw = 1

    im4_d = ax4.plot(result_LaxW_4[0], H+result_LaxW_4[1][int(t1/dt4)-1], 'r-', label='CFL = 2.0', linewidth=lw, alpha=0.2)
    im4_e = ax4.plot(result_LaxW_1[0], H+result_LaxW_1[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im4_f = ax4.plot(result_UP_1[0], H+res_exact_t1, label='Exact Solution', linewidth=lw)
    im4_a = ax4.plot(result_LaxW_1[0], H+result_LaxW_1[1][int(t1/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im4_b = ax4.plot(result_LaxW_2[0], H+result_LaxW_2[1][int(t1/dt2)-1], label='CFL = 0.9', linewidth=lw)
    im4_c = ax4.plot(result_LaxW_3[0], H+result_LaxW_3[1][int(t1/dt3)-1], label='CFL = 1.0', linewidth=lw)

    im5_d = ax5.plot(result_LaxW_4[0], H+result_LaxW_4[1][int(t2/dt4)-1], 'r-', label='CFL = 2.0', linewidth=lw, alpha=0.2)   
    im5_e = ax5.plot(result_LaxW_1[0], H+result_LaxW_1[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im5_f = ax5.plot(result_UP_1[0], H+res_exact_t2, label='Exact Solution', linewidth=lw)
    im5_a = ax5.plot(result_LaxW_1[0], H+result_LaxW_1[1][int(t2/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im5_b = ax5.plot(result_LaxW_2[0], H+result_LaxW_2[1][int(t2/dt2)-1], label='CFL = 0.9', linewidth=lw)
    im5_c = ax5.plot(result_LaxW_3[0], H+result_LaxW_3[1][int(t2/dt3)-1], label='CFL = 1.0',linewidth=lw)

    im6_d = ax6.plot(result_LaxW_4[0], H+result_LaxW_4[1][int(t3/dt4)-1], 'r-', label='CFL = 2.0', linewidth=lw, alpha=0.2)
    im6_e = ax6.plot(result_LaxW_1[0], H+result_LaxW_1[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im6_f = ax6.plot(result_UP_1[0], H+res_exact_t3, label='Exact Solution', linewidth=lw)
    im6_a = ax6.plot(result_LaxW_1[0], H+result_LaxW_1[1][int(t3/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im6_b = ax6.plot(result_LaxW_2[0], H+result_LaxW_2[1][int(t3/dt2)-1], label='CFL = 0.9', linewidth=lw)
    im6_c = ax6.plot(result_LaxW_3[0], H+result_LaxW_3[1][int(t3/dt3)-1], label='CFL = 1.0', linewidth=lw)

    ax4.set_title('t = 0.5s')
    ax4.set_ylim([H-A, H+A])
    ax4.set_xlabel('Length (m)')
    ax4.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax4.legend()
    ax4.grid()

    ax5.set_title('t = 1.0s')
    ax5.set_ylim([H-A, H+A])
    ax5.set_xlabel('Length (m)')
    ax5.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax5.legend()
    ax5.grid()

    ax6.set_title('t = 2.0s')
    ax6.set_ylim([H-A, H+A])
    ax6.set_xlabel('Length (m)')
    ax6.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax6.legend()
    ax6.grid()

    fig2.suptitle('Lax-Wendroff ' + r'$\lambda$' + ' = 12m')


    # Upwind plotting, lambda = 6m
    fig3, (ax7, ax8, ax9) = plt.subplots(3,1)
    lw = 1

    im7_e = ax7.plot(result_UP_1_hf[0], H+result_UP_1_hf[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im7_f = ax7.plot(result_UP_1_hf[0], H+res_exact_t1_hf, label='Exact Solution', linewidth=lw)
    im7_a = ax7.plot(result_UP_1_hf[0], H+result_UP_1_hf[1][int(t1/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im7_b = ax7.plot(result_UP_2_hf[0], H+result_UP_2_hf[1][int(t1/dt2)-1], label='CFL = 0.9', linewidth=lw)
 
    im8_e = ax8.plot(result_UP_1_hf[0], H+result_UP_1_hf[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im8_f = ax8.plot(result_UP_1_hf[0], H+res_exact_t2_hf, label='Exact Solution', linewidth=lw)
    im8_a = ax8.plot(result_UP_1_hf[0], H+result_UP_1_hf[1][int(t2/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im8_b = ax8.plot(result_UP_2_hf[0], H+result_UP_2_hf[1][int(t2/dt2)-1], label='CFL = 0.9', linewidth=lw)

    im9_e = ax9.plot(result_UP_1_hf[0], H+result_UP_1_hf[1][0], 'm--', label='Initial Condition', linewidth=lw, alpha=0.7)
    im9_f = ax9.plot(result_UP_1_hf[0], H+res_exact_t3_hf, label='Exact Solution', linewidth=lw)
    im9_a = ax9.plot(result_UP_1_hf[0], H+result_UP_1_hf[1][int(t3/dt1)-1], label='CFL = 0.5', linewidth=lw)
    im9_b = ax9.plot(result_UP_2_hf[0], H+result_UP_2_hf[1][int(t3/dt2)-1], label='CFL = 0.9', linewidth=lw)

    ax7.set_title('t = 0.5s')
    ax7.set_ylim([H-A, H+A])
    ax7.set_xlabel('Length (m)')
    ax7.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax7.legend()
    ax7.grid()

    ax8.set_title('t = 1.0s')
    ax8.set_ylim([H-A, H+A])
    ax8.set_xlabel('Length (m)')
    ax8.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax8.legend()
    ax8.grid()

    ax9.set_title('t = 2.0s')
    ax9.set_ylim([H-A, H+A])
    ax9.set_xlabel('Length (m)')
    ax9.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax9.legend()
    ax9.grid()

    fig3.suptitle('Upwind, ' + r'$\lambda$' + ' = 6m')
    
    fig4, ax10 = plt.subplots()

    im10_a = ax10.plot(result_LaxW_periodic[0], H+result_LaxW_periodic[1][int(50/dt3)-1], label='Lax-W, CFL = 1.0 after 5*Lx')
    im10_b = ax10.plot(result_LaxW_periodic[0], H+result_LaxW_periodic[1][0], label="Exact Solution after 5*Lx")
    ax10.set_title('Lax-W with Periodic BCs')
    ax10.set_xlabel('Length (m)')
    ax10.set_ylabel('H + ' + r'$\eta$' + ' (m)')
    ax10.legend()
    ax10.grid()
    plt.show()

    # commented out because the gif takes a long time to compile and generates 1200 images
    # fn.create_GIF(result_LaxW_periodic[0], result_LaxW_periodic[1], result_LaxW_periodic[2], H)

if __name__ == '__main__':
    main()