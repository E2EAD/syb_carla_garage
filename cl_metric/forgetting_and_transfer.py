import numpy as np
import pandas as pd


def calculate_metrics(skill_matrix, learning_order):
    """
    Calculate forgetting, forward transfer, and backward transfer metrics based on the skill matrix.

    Parameters:
    skill_matrix (numpy.array): A 5x5 matrix where rows represent the learning state
                               (after learning the first i+1 tasks) and columns represent
                               the test skills.
                               Rows: After learning [EmergencyBrake],
                                     [EmergencyBrake, TrafficSign],
                                     [EmergencyBrake, TrafficSign, Merging],
                                     [EmergencyBrake, TrafficSign, Merging, Overtaking],
                                     [All 5 tasks]
                               Columns: [Merging, Overtaking, EmergencyBrake, GiveWay, TrafficSign]

    Returns:
    tuple: (forgetting_list, forgetting_avg,
            forward_transfer_list, forward_transfer_avg,
            backward_transfer_list, backward_transfer_avg)
    """
    # 定义任务名称到测试索引的映射
    # 测试顺序(列索引):
    # 0: Merging, 1: Overtaking, 2: EmergencyBrake, 3: GiveWay, 4: TrafficSign
    task_to_test_index = {
        'EmergencyBrake': 2,
        'TrafficSign': 4,
        'Merging': 0,
        'Overtaking': 1,
        'GiveWay': 3
    }

    N = skill_matrix.shape[0]  # 任务数量

    # 确保数据在[0,1]范围内（如果输入是百分比则除以100）
    if np.max(skill_matrix) > 1.0:
        print("警告: 检测到输入数据可能为百分比，已自动归一化到[0,1]范围")
        skill_matrix = skill_matrix / 100.0

    # 计算遗忘率（每个任务）
    forgetting_list = []
    for i in range(N - 1):  # 排除最后一个任务
        task_name = learning_order[i]
        test_index = task_to_test_index[task_name]
        # print(task_name)

        # P_i(Δi): 任务i完成训练后的成功率
        p_i_delta_i = skill_matrix[i, test_index]

        # P_i(T): 所有任务训练完成后的最终成功率
        p_i_T = skill_matrix[N - 1, test_index]

        # print(p_i_delta_i, p_i_T)
        # F_i = P_i(Δi) - P_i(T)
        f_i = p_i_delta_i - p_i_T
        forgetting_list.append(f_i)

    # 为最后一个任务添加np.nan占位
    forgetting_list.append(np.nan)

    # F = (1/(N-1)) * Σ(F_i) - 只考虑有值的部分
    valid_forgetting = [f for f in forgetting_list if not np.isnan(f)]
    forgetting_avg = np.mean(valid_forgetting) if valid_forgetting else np.nan

    # 计算前向迁移率（每个任务）
    forward_transfer_list = [np.nan]  # 第一个任务没有前向迁移率，用np.nan占位

    for i in range(1, N):  # 从第二个任务开始
        # 前i个任务的平均成功率
        sum_prev = 0
        for k in range(i):
            task_name = learning_order[k]
            test_index = task_to_test_index[task_name]
            sum_prev += skill_matrix[k, test_index]

        # FT_i = (1/i) * Σ(P_k(Δk))
        ft_i = sum_prev / i
        forward_transfer_list.append(ft_i)

    # FT = (1/(N-1)) * Σ(FT_i) - 只考虑有值的部分
    valid_forward_transfer = [ft for ft in forward_transfer_list if not np.isnan(ft)]
    forward_transfer_avg = np.mean(valid_forward_transfer) if valid_forward_transfer else np.nan

    # 计算后向迁移率（每个任务）
    backward_transfer_list = []
    for i in range(N - 1):  # 排除最后一个任务（它没有后续任务）
        # 计算第i个任务对后续任务的影响
        sum_next = 0
        count = 0

        for k in range(i + 1, N):  # 从i+1到N-1
            task_name = learning_order[k]
            test_index = task_to_test_index[task_name]
            # Pk(Δi): 在完成第i个任务训练后，测试第k个任务的成功率
            p_k_delta_i = skill_matrix[i, test_index]
            sum_next += p_k_delta_i
            count += 1

        # BTi = (1/count) * Σ(Pk(Δi))
        bt_i = sum_next / count if count > 0 else 0
        backward_transfer_list.append(bt_i)

    # 为最后一个任务添加np.nan占位
    backward_transfer_list.append(np.nan)

    # BT = (1/(N-1)) * Σ(BTi) - 只考虑有值的部分
    valid_backward_transfer = [bt for bt in backward_transfer_list if not np.isnan(bt)]
    backward_transfer_avg = np.mean(valid_backward_transfer) if valid_backward_transfer else np.nan

    return forgetting_list, forgetting_avg, forward_transfer_list, forward_transfer_avg, backward_transfer_list, backward_transfer_avg


if __name__ == '__main__':
    # 假设您已经将表格读取为numpy数组
    skill_matrix = [[43.75, 15.56, 71.67, 50, 68.95],
                    [46.25,15.56,81.67,40,80],
                    [50,15.56,51.67,60,75.26],
                    [45,31.11,31.67,60,63.68],
                    [22.5,13.33,20,40,45.79]]

    skill_matrix = np.array(skill_matrix)  # 您的实际数据
    learning_order = ['EmergencyBrake', 'TrafficSign', 'Merging', 'Overtaking', 'GiveWay']

    # 计算指标
    forgetting_list, forgetting_avg, forward_transfer_list, forward_transfer_avg, backward_transfer_list, backward_transfer_avg = calculate_metrics(
        skill_matrix, learning_order)

    # 准备结果数据 - 详细列出每个任务的指标
    results = []

    # 1. 遗忘率（最后一个任务为np.nan）
    results.append({'Task': 'Overall', 'Metric': 'Forgetting', 'Value': forgetting_avg})
    for i, task_name in enumerate(learning_order):
        results.append({'Task': task_name, 'Metric': 'Forgetting', 'Value': forgetting_list[i]})

    # 2. 前向迁移率（第一个任务为np.nan）
    results.append({'Task': 'Overall', 'Metric': 'Forward Transfer', 'Value': forward_transfer_avg})
    for i, task_name in enumerate(learning_order):
        results.append({'Task': task_name, 'Metric': 'Forward Transfer', 'Value': forward_transfer_list[i]})

    # 3. 后向迁移率（最后一个任务为np.nan）
    results.append({'Task': 'Overall', 'Metric': 'Backward Transfer', 'Value': backward_transfer_avg})
    for i, task_name in enumerate(learning_order):
        results.append({'Task': task_name, 'Metric': 'Backward Transfer', 'Value': backward_transfer_list[i]})

    # 4. 任务性能快照 - 显示每个任务在完成时的性能
    for i, task_name in enumerate(learning_order):
        test_index = {'EmergencyBrake': 2, 'TrafficSign': 4, 'Merging': 0,
                      'Overtaking': 1, 'GiveWay': 3}[task_name]
        performance = skill_matrix[i, test_index]
        results.append({'Task': task_name, 'Metric': 'Performance at Completion', 'Value': performance})

    # 5. 任务最终性能 - 显示每个任务在所有任务完成后的性能
    for i, task_name in enumerate(learning_order):
        test_index = {'EmergencyBrake': 2, 'TrafficSign': 4, 'Merging': 0,
                      'Overtaking': 1, 'GiveWay': 3}[task_name]
        final_performance = skill_matrix[-1, test_index]
        results.append({'Task': task_name, 'Metric': 'Final Performance', 'Value': final_performance})

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 按任务和指标排序，使结果更清晰
    metric_order = ['Forgetting', 'Forward Transfer', 'Backward Transfer',
                    'Performance at Completion', 'Final Performance']
    df['Metric'] = pd.Categorical(df['Metric'], categories=metric_order, ordered=True)
    df = df.sort_values(['Task', 'Metric'])

    # 保存结果到CSV
    df.to_csv('driving_skills_metrics.csv', index=False)

    # 打印详细结果
    print("\n===== 详细指标结果 =====")
    print(df.to_string(index=False))

    # 打印关键总结
    print("\n===== 指标总结 =====")
    print(f"整体遗忘率 ([-1,1], 越低越好): {forgetting_avg:.4f}")
    print(f"整体前向迁移率 ([0,1], 越高越好): {forward_transfer_avg:.4f}")
    print(f"整体后向迁移率 ([0,1], 越高越好): {backward_transfer_avg:.4f}")

    print("\n===== 每个任务的具体指标 =====")
    print(f"遗忘率 (最后一个任务为NaN):")
    for i, task_name in enumerate(learning_order):
        value = forgetting_list[i]
        value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
        print(f"  {task_name}: {value_str}")

    print(f"\n前向迁移率 (第一个任务为NaN):")
    for i, task_name in enumerate(learning_order):
        value = forward_transfer_list[i]
        value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
        print(f"  {task_name}: {value_str}")

    print(f"\n后向迁移率 (最后一个任务为NaN):")
    for i, task_name in enumerate(learning_order):
        value = backward_transfer_list[i]
        value_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
        print(f"  {task_name}: {value_str}")

    print("\n详细结果已保存至 'driving_skills_metrics.csv'")
