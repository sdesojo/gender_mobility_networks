# -- ANONYMIZE SAMPLED GRAPHS
dic_M_graph_anonymous = {}
for i, k in enumerate(dic_M_graph.keys()):
    k_anonym = f"user_{i}"
    dic_M_graph_anonymous[k_anonym] = dic_M_graph[k].copy()

dic_F_graph_anonymous = {}
for i, k in enumerate(dic_F_graph.keys()):
    k_anonym = f"user_{i}"
    dic_F_graph_anonymous[k_anonym] = dic_F_graph[k].copy()

# Save the anonymized graphs
with open(input_fig_path + "nw_M_20k_anonym.pkl", "wb") as f:
    pickle.dump(dic_M_graph_anonymous, f)

with open(input_fig_path + "nw_F_20k_anonym.pkl", "wb") as f:
    pickle.dump(dic_F_graph_anonymous, f)


# -- Get Share of countries with significant results

dic_SIGNF = {
    mlab: {test: {"sig_MALE": 0, "sig_FEMALE": 0, "sig_None": 0} for test in TEST}
    for mlab in NW_MLAB
}
for mlab in NW_MLAB:
    for test in TEST:
        for i, ctry in enumerate(CTRY):
            res_true = dic_RES[ctry][test]["True"]
            res_shuf = dic_RES[ctry][test]["Shuffled"]

            data = 2 * res_true["rel_dif_" + mlab]["mean"]
            shuf_comb = [2 * e["rel_dif_" + mlab]["mean"] for e in res_shuf]
            p5, q1, q3, p95 = np.percentile(shuf_comb, [5, 25, 75, 95])
            # print(mlab, test, ctry, data, p5, q1, q3, p95)

            if data > 0 and data > p95:
                dic_SIGNF[mlab][test]["sig_MALE"] += 1

            elif data < 0 and data < p5:
                dic_SIGNF[mlab][test]["sig_FEMALE"] += 1

            else:
                dic_SIGNF[mlab][test]["sig_None"] += 1

print(mlab, ctry)
dic_SIGNF[mlab]["all"]
