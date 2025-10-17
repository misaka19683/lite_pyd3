# import sympy
from sympy import *
# from sympy import Eq
sp0,sp1,sp2,sp3,sp4,sp5,sp6=symbols('s0+ s1+ s2+ s3+ s4+ s5+ s6+')
sm0,sm1,sm2,sm3,sm4,sm5,sm6=symbols('s0- s1- s2- s3- s4- s5- s6-')
# sympy.Eq
term1=(sp0-sp1)*(sp1-sp2)*(sp2-sp3)*(sp3-sp4)*(sp4-sp5)*(sp5-sp6)
# term2=(sm0-sm1)*(sm1-sm2)*(sm2-sp3)*(sm3-sm4)*(sm4-sm5)*(sm5-sp6)
term2=(sm0-sm1)*(sm1-sm2)*(sm2-sm3)*(sm3-sm4)*(sm4-sm5)*(sm5-sm6)
# print(expand(term1))
# print(expand(term2))

# 展开
expanded_term1 = expand(term1)
expanded_term2 = expand(term2)

# 应用 (S_i^+)^2 = 0 和 (S_i^-)^2 = 0
replacements = {
    sp0**2: 0, sp1**2: 0, sp2**2: 0, sp3**2: 0, sp4**2: 0, sp5**2: 0, sp6**2: 0,
    sm0**2: 0, sm1**2: 0, sm2**2: 0, sm3**2: 0, sm4**2: 0, sm5**2: 0, sm6**2: 0
}
# 简化
simplified_term1 = expanded_term1.subs(replacements)
simplified_term2 = expanded_term2.subs(replacements)

print(f"term1={simplified_term1}")
print(f"term2={simplified_term2}")

# combined = simplified_term1 + simplified_term2
#
# # 定义厄米共轭映射
# dagger_map = {sp0: sm0, sp1: sm1, sp2: sm2, sp3: sm3, sp4: sm4, sp5: sm5, sp6: sm6,
#               sm0: sp0, sm1: sp1, sm2: sp2, sm3: sp3, sm4: sp4, sm5: sp5, sm6: sp6}
#
# for terms in combined.as_order_terms():
#     coeff,ops=terms.as_coeff_Mul()
#     new_ops=1
#     for factor in ops.as_ordered_factors():
#         new_ops*=dagger_map[factor]
    # combined_dagg

# 调查如果使用sympy完成我的计算，目前需要确认这两项相加之后是否是厄米共轭的东西
# 因此需要sympy能够忠实地反应自旋算符之类的量子力学符号